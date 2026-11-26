#!/usr/bin/env python3
"""Proof impact of a Rust change: which verified functions it touches.

The proofs are hand-written Lean models of the Rust code, so a change to a
proven function can leave its model, and every theorem about it, describing
code that no longer exists. This script finds those functions from a git diff
and the COVERAGE.tsv rows, without building any Lean.

Usage:
  proofs/pr_impact.py --repo <checkout> --base <sha> --head <sha>
                      [--proofs <proofs dir>] [--json OUT] [--md OUT]

--repo is a checkout holding both commits (graph/src and src are read from
git, not from the work tree). --proofs defaults to this script's directory, so
the proofs may come from another branch than the code under review.

Reported:
  changed   production fns whose body the diff touches and that have a
            PROVEN / CONDITIONAL / MODELLED row: their model may be stale
  added     new production fns no COVERAGE.tsv row lists
  removed   fns the diff deletes or renames that still have rows
A fn spans from its `fn` line (or the topmost attribute above it that changes
behaviour, such as `#[cfg]`) to its closing brace, so a diff that only adds a
lint or inlining attribute such as `#[allow(..)]`, or a doc comment, touches no fn.
Fns are matched to rows by (file, name), the row's line only breaking ties
between same-name fns of one file.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TOPS = ('graph/src/', 'src/')
CHECKED = ('PROVEN', 'CONDITIONAL', 'MODELLED')
HUNK = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')


def git(repo, *args):
    return subprocess.run(['git', '-C', repo, *args], check=True,
                          capture_output=True, text=True).stdout


def tree(repo, sha, root):
    """graph/src and src of `sha`, extracted under `root`, so coverage.sh sees
    each file in its crate (an out-of-line `#[cfg(test)] mod x;` lives in the
    parent file, not in x.rs)."""
    archive = subprocess.run(['git', '-C', repo, 'archive', sha, '--', *TOPS],
                             check=True, capture_output=True).stdout
    subprocess.run(['tar', '-x', '-C', root], input=archive, check=True)
    return root


def fns_of(proofs, root, rel):
    """[(line, name, is_test)] for one Rust file of an extracted tree, via
    coverage.sh --fns, so test detection matches the coverage gate exactly."""
    if not os.path.exists(os.path.join(root, rel)):
        return []
    out = subprocess.run([os.path.join(proofs, 'coverage.sh'), '--fns', rel],
                         env={**os.environ, 'COVERAGE_RUST_ROOT': root},
                         check=True, capture_output=True, text=True).stdout
    res = []
    for l in out.splitlines():
        line, name, kind = l.split('\t')
        res.append((int(line), name, kind == 'test'))
    return res


def read_lines(root, rel):
    """The lines of one Rust file of an extracted tree ([] if it is absent)."""
    p = os.path.join(root, rel)
    if not os.path.exists(p):
        return []
    with open(p, encoding='utf-8', errors='replace') as f:
        return f.read().split('\n')


# Attributes that leave what a fn computes unchanged (lints, docs, inlining
# hints): editing one above a fn touches no fn. Any other attribute (`cfg`,
# `cfg_attr`, `test`, `no_mangle`, `target_feature`, ...) can change which body
# is compiled or how it runs, so it belongs to the fn below it.
COSMETIC_ATTRS = frozenset(('allow', 'warn', 'expect', 'deny', 'forbid', 'doc',
                            'must_use', 'inline', 'cold', 'rustfmt::skip'))
ATTR_PATH = re.compile(r'#!?\[\s*([A-Za-z_][A-Za-z0-9_]*(?:\s*::\s*[A-Za-z_][A-Za-z0-9_]*)*)')


RAW_OPEN = re.compile(r'(?<![A-Za-z0-9_])[bc]?r(#*)"')


def lex(lines):
    """(code, cont) for a Rust file: `code[i]` is line i+1 with comments removed
    and string and char literals blanked to `""` / `''`, and `cont[i]` is True
    when line i+1 starts inside a string or block comment. The state carries
    across lines, so a raw string, a multi-line string or a `/* */` comment can
    never pass its `[`, `}` or `"` off as code."""
    code, cont = [], []
    mode, hashes, depth = None, 0, 0             # mode: None, 'str', 'raw', 'block'
    for line in lines:
        cont.append(mode is not None)
        out, i, n = [], 0, len(line)
        while i < n:
            if mode == 'block':
                if line.startswith('/*', i):
                    depth, i = depth + 1, i + 2
                elif line.startswith('*/', i):
                    depth, i = depth - 1, i + 2
                    if not depth:
                        mode = None
                        out.append(' ')
                else:
                    i += 1
            elif mode == 'str':
                if line[i] == '\\':
                    i += 2
                else:
                    mode = None if line[i] == '"' else mode
                    i += 1
            elif mode == 'raw':
                if line.startswith('"' + '#' * hashes, i):
                    mode, i = None, i + 1 + hashes
                else:
                    i += 1
            elif line.startswith('//', i):
                break
            elif line.startswith('/*', i):
                mode, depth, i = 'block', 1, i + 2
            elif m := RAW_OPEN.match(line, i):
                out.append('""')
                mode, hashes, i = 'raw', len(m[1]), m.end()
            elif line[i] == '"':
                out.append('""')
                mode, i = 'str', i + 1
            elif line[i] == "'" and i + 1 < n and line[i + 1] == '\\':
                j = line.find("'", i + 3)                 # '\n', '\'', '\u{7d}'
                out.append("''")
                i = (j if j >= 0 else n) + 1
            elif line[i] == "'" and i + 2 < n and line[i + 2] == "'":
                out.append("''")                          # 'x'; otherwise a lifetime
                i += 3
            else:
                out.append(line[i])
                i += 1
        code.append(''.join(out).rstrip())
    return code, cont


def attr_start(code, j):
    """1-based line of the `#[` that opens an attribute ending on line `j`
    (possibly multi-line, possibly with blank lines inside), or None when line `j` does not close an attribute
    that fills its lines. `code` is lex(lines)[0]."""
    for k in range(j, 0, -1):
        s = code[k - 1].strip()
        if s.startswith(('#[', '#![')):
            text = '\n'.join(code[i - 1].strip() for i in range(k, j + 1))
            depth = 0
            for n, ch in enumerate(text):
                depth += (ch == '[') - (ch == ']')
                if depth == 0 and ch == ']':
                    # the attribute's own `]` must be the last character
                    return k if n == len(text) - 1 else None
            return None
        if s.endswith((';', '{', '}')):
            return None                           # a blank line may sit inside the attribute
    return None


def is_cosmetic(code, k):
    """True when the attribute opening on line `k` is one of COSMETIC_ATTRS."""
    m = ATTR_PATH.match(code[k - 1].strip())
    return bool(m) and re.sub(r'\s+', '', m[1]) in COSMETIC_ATTRS


def lead_of(lines, code, start):
    """(first, behavioural) for the attributes, comments and blank lines directly
    above the item on line `start`: `first` is the lead's first line, and
    `behavioural` the lines of its attributes outside COSMETIC_ATTRS, which
    belong to the item below; the rest of the lead belongs to no fn. A line
    holding code beside a comment or attribute (`/* a */ let x = 1;`) ends the
    lead."""
    j, behavioural = start - 1, set()
    while j >= 1:
        s = code[j - 1].strip()
        if not s:
            j -= 1                                # blank, or only comments
        elif s.endswith(']') and (k := attr_start(code, j)) is not None:
            if code[k - 1].strip().startswith('#!['):
                break                             # an inner attribute: it is the enclosing item's
            if not is_cosmetic(code, k):
                behavioural.update(range(k, j + 1))
            j = k - 1
        else:
            break
    return j + 1, frozenset(behavioural)


def indent(line):
    return len(line) - len(line.lstrip())


ITEM = re.compile(r'(pub(\([^)]*\))?\s+)?(unsafe\s+)?'
                  r'(mod|impl|struct|enum|union|trait|const|static|type|use|extern|macro_rules!)\b')


def close_of(code, start, name):
    """Line of the `}` closing the fn `name` declared on line `start`, by brace
    matching over lexed code (lex()[0]). In the signature `()`, `[]`, `<>` and
    braces nested in them (`[u8; { 1 }]`, `Foo<{ N }>`) are skipped, so the
    first `{` outside them opens the body, and a `;` before it ends a bodyless
    fn. None when the file ends first."""
    m = re.search(r'\bfn\s+' + re.escape(name) + r'\b', code[start - 1])
    pos = m.end() if m else 0
    sig, braces, body, prev = 0, 0, 0, ''
    for k in range(start, len(code) + 1):
        for ch in (code[k - 1][pos:] if k == start else code[k - 1]):
            if body:
                if ch == '{':
                    body += 1
                elif ch == '}':
                    body -= 1
                    if not body:
                        return k
            elif braces:                          # a const block inside the signature
                braces += (ch == '{') - (ch == '}')
            elif ch == '{':
                if sig:
                    braces = 1
                else:
                    body = 1
            elif ch in '([<':
                sig += 1
            elif ch in ')]' or (ch == '>' and prev not in '-='):
                sig = max(sig - 1, 0)
            elif ch == ';' and not sig:
                return k                          # a bodyless trait or extern fn
            prev = ch
    return None


def ends_of(fns, lines):
    """(lead, first, last, behavioural) for each fn: its body is [first, last]
    and [lead, first) is the lead above it, of which only the `behavioural`
    lines (attributes that can change behaviour: `#[cfg]`, `#[target_feature]`,
    ...) belong to the fn. `first` is the fn line and `last` its closing brace
    (when the braces never balance, the line before the lead of the next item
    at the fn's indent or shallower, or the end of the file). Comments, blank
    lines and lint attributes between two fns therefore belong to neither. A
    `fn` inside a string or comment owns nothing."""
    code, cont = lex(lines)
    spans = []
    for i, (start, name, _) in enumerate(fns):
        if not re.search(r'\bfn\s+' + re.escape(name) + r'\b', code[start - 1]):
            spans.append((start, start, start - 1, frozenset()))   # not a real fn: owns nothing
            continue
        lead, behavioural = lead_of(lines, code, start)
        last = close_of(code, start, name)
        if last is None:
            base = indent(lines[start - 1])
            last = len(lines)
            for k in range(start + 1, len(lines) + 1):
                s = code[k - 1].strip()
                if not s or cont[k - 1] or indent(lines[k - 1]) > base:
                    continue
                if any(f[0] == k for f in fns[i + 1:]) or ITEM.match(s):
                    last = lead_of(lines, code, k)[0] - 1
                    break
        spans.append((lead, start, max(start, last), behavioural))
    return spans


def owner(fns, ends, line):
    """Index of the innermost fn whose span holds `line`, or None when none
    does (between fns, or in the comments and lint attributes above any fn,
    nested ones included). Innermost, so code after a nested fn still belongs
    to the outer fn."""
    best = None
    for i, (start, _, _) in enumerate(fns):
        lead, first, last, behavioural = ends[i]
        if lead <= line < first:
            # a doc comment or lint attribute belongs to no fn, even on a nested fn
            return i if line in behavioural else None
        if first <= line <= last and (best is None or start > fns[best][0]):
            best = i
    return best


def load_rows(proofs):
    rows = defaultdict(list)  # (file, fn) -> [row]
    for tsv in sorted(glob.glob(os.path.join(proofs, '*', 'COVERAGE.tsv'))):
        project = os.path.basename(os.path.dirname(tsv))
        for n, l in enumerate(open(tsv), 1):
            if l.startswith('#') or l.startswith('file\t') or not l.strip():
                continue
            f = l.rstrip('\n').split('\t')
            if len(f) < 4:
                continue
            bucket = f[3].strip().upper().replace('_', ' ')
            # bare identifier, normalised exactly as coverage.sh's resolve():
            # `CondTraverseOp::new` -> `new`, `a/b` -> `a`, `T.f` -> `f`, `f(x)` -> `f`
            name = f[1].strip().split('/')[0].split('::')[-1].split('.')[-1].split('(')[0].strip()
            rows[(f[0].strip().lstrip('./'), name)].append(dict(project=project, line=int(f[2]) if f[2].isdigit() else 0,
                                           bucket=bucket, lean=f[4] if len(f) > 4 else '',
                                           tsv=f'proofs/{project}/COVERAGE.tsv:{n}'))
    return rows


def pick(rows, line, same_name):
    """Rows for one fn; with several same-name fns in a file keep the rows
    nearest its base line."""
    if len(same_name) <= 1 or not rows:
        return rows
    near = min(abs(r['line'] - line) for r in rows)
    return [r for r in rows if abs(r['line'] - line) == near]


def show(repo, sha, rel):
    try:
        return git(repo, 'show', f'{sha}:{rel}')
    except subprocess.CalledProcessError:
        return None


def analyse(repo, base, head, proofs):
    with tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as h:
        return _analyse(repo, base, head, proofs, tree(repo, base, b), tree(repo, head, h))


def _analyse(repo, base, head, proofs, base_root, head_root):
    rows = load_rows(proofs)
    names = git(repo, 'diff', '--name-status', '-M', base, head, '--', *TOPS).splitlines()
    changed, added, removed = [], [], []
    for entry in names:
        parts = entry.split('\t')
        status, paths = parts[0], parts[1:]
        old_rel, new_rel = paths[0], paths[-1]
        if not new_rel.endswith('.rs') and not old_rel.endswith('.rs'):
            continue
        old_fns = fns_of(proofs, base_root, old_rel) if status[0] != 'A' else []
        new_fns = fns_of(proofs, head_root, new_rel) if status[0] != 'D' else []
        old_ends = ends_of(old_fns, read_lines(base_root, old_rel))
        new_ends = ends_of(new_fns, read_lines(head_root, new_rel))
        diff = git(repo, 'diff', '-U0', '-M', base, head, '--', *dict.fromkeys([old_rel, new_rel]))
        touched_new, touched_old = set(), set()
        for l in diff.splitlines():
            m = HUNK.match(l)
            if not m:
                continue
            os_, oc, ns, nc = int(m[1]), int(m[2] or 1), int(m[3]), int(m[4] or 1)
            for ln in range(ns, ns + nc):
                i = owner(new_fns, new_ends, ln)
                if i is not None:
                    touched_new.add(i)
            for ln in range(os_, os_ + oc):   # a pure insertion (-N,0) removes no old line
                i = owner(old_fns, old_ends, ln)
                if i is not None:
                    touched_old.add(i)
        old_names = defaultdict(list)
        for i, (ln, nm, t) in enumerate(old_fns):
            old_names[nm].append(i)
        new_names = defaultdict(list)
        for i, (ln, nm, t) in enumerate(new_fns):
            new_names[nm].append(i)

        # A body edit shows on the new side; a deletion inside a fn only on the old.
        seen = set()
        for i in sorted(touched_new):
            ln, nm, test = new_fns[i]
            if test or nm not in old_names:
                continue
            k = new_names[nm].index(i)
            oi = old_names[nm][min(k, len(old_names[nm]) - 1)]
            seen.add(oi)
            r = [x for x in pick(rows.get((old_rel, nm), []), old_fns[oi][0], old_names[nm]) if x['bucket'] in CHECKED]
            if r:
                changed.append(dict(file=new_rel, fn=nm, line=ln, rows=r))
        for oi in sorted(touched_old - seen):
            ln, nm, test = old_fns[oi]
            if test or nm not in new_names:
                continue
            k = old_names[nm].index(oi)
            ni = new_names[nm][min(k, len(new_names[nm]) - 1)]
            r = [x for x in pick(rows.get((old_rel, nm), []), ln, old_names[nm]) if x['bucket'] in CHECKED]
            if r and not any(c['fn'] == nm and c['file'] == new_rel for c in changed):
                changed.append(dict(file=new_rel, fn=nm, line=new_fns[ni][0], rows=r))
        for nm, idx in new_names.items():
            extra = len(idx) - len(old_names.get(nm, []))
            prod = [i for i in idx if not new_fns[i][2]]
            # a new fn sharing a name with a listed one is still unlisted unless there is a
            # row for each production occurrence
            if extra > 0 and prod and len(rows.get((new_rel, nm), [])) < len(prod):
                for i in idx[-extra:]:
                    if not new_fns[i][2]:
                        added.append(dict(file=new_rel, fn=nm, line=new_fns[i][0]))
        for nm, idx in old_names.items():
            if len(idx) > len(new_names.get(nm, [])) and rows.get((old_rel, nm)):
                removed.append(dict(file=old_rel, fn=nm, line=old_fns[idx[0]][0],
                                    rows=rows[(old_rel, nm)]))
    return dict(base=base, head=head, changed=changed, added=added, removed=removed)


def markdown(res, proofs_ref):
    c, a, r = res['changed'], res['added'], res['removed']
    out = ['<!-- proof-impact -->', '### Proof impact', '']
    if not (c or a or r):
        out.append('No verified function is touched, and every new production function '
                   'is already listed. Nothing to re-prove.')
    else:
        out.append(f'Against the Lean proofs at `{proofs_ref}`. The proofs are models of the Rust '
                   'code, so a changed function\'s model and theorems may now describe the old code.')
    if c:
        out += ['', f'**{len(c)} verified function(s) changed** — re-check their models:', '',
                '| function | bucket | Lean project | theorems / model |', '|---|---|---|---|']
        for x in c:
            for row in x['rows']:
                lean = row['lean'].replace('|', '\\|')
                if len(lean) > 160:
                    lean = lean[:157] + '…'
                out.append(f"| `{x['file']}` `{x['fn']}` | {row['bucket']} | `{row['project']}` | {lean} |")
    if r:
        out += ['', f'**{len(r)} function(s) with coverage rows removed or renamed:** ' +
                ', '.join(f"`{x['file']}` `{x['fn']}`" for x in r)]
    if a:
        out += ['', f'**{len(a)} new production function(s) with no coverage row:** ' +
                ', '.join(f"`{x['file']}` `{x['fn']}`" for x in a)]
    out += ['', '<sub>Advisory, not part of ci-gate. Generated by `proofs/pr_impact.py`.</sub>']
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--head', required=True)
    ap.add_argument('--proofs', default=HERE)
    ap.add_argument('--proofs-ref', default='proofs/lean4-verification')
    ap.add_argument('--json')
    ap.add_argument('--md')
    a = ap.parse_args()
    base = git(a.repo, 'merge-base', a.base, a.head).strip()
    res = analyse(a.repo, base, a.head, a.proofs)
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)
    md = markdown(res, a.proofs_ref)
    if a.md:
        open(a.md, 'w').write(md)
    else:
        sys.stdout.write(md)


if __name__ == '__main__':
    main()
