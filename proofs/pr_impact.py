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
A fn spans from its `fn` line to the line before the next fn's leading
attributes, comments and blank lines (or to the end of the file), so a diff that
only adds `#[allow(..)]` or a doc comment above a fn touches no fn; fns are matched to rows by (file, name), the row's line only breaking ties
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


def attr_start(lines, j):
    """1-based line of the `#[` that opens a multi-line attribute ending on line
    `j` (`)]`), or None when line `j` does not close an attribute."""
    for k in range(j, 0, -1):
        s = lines[k - 1].strip()
        if s.startswith(('#[', '#![')):
            text = ' '.join(lines[i - 1].strip() for i in range(k, j + 1))
            depth = 0
            for n, ch in enumerate(text):
                depth += (ch == '[') - (ch == ']')
                if depth == 0 and ch == ']':
                    # the attribute's own `]` must be the last character
                    return k if n == len(text) - 1 else None
            return None
        if not s or s.endswith((';', '{', '}')):
            return None
    return None


def lead_start(lines, start):
    """First line of the attributes, comments and blank lines directly above the
    `fn` on line `start`: lines that belong to no fn's body."""
    j = start - 1
    while j >= 1:
        s = lines[j - 1].strip()
        if not s or s.startswith('//') or (s.startswith(('#[', '#![')) and s.endswith(']')):
            j -= 1
        elif s.endswith('*/') and ('/*' not in s or s.startswith('/*')):
            # a block comment, possibly multi-line: back to the line it opens on
            k = j
            while k >= 1 and not lines[k - 1].strip().startswith('/*'):
                if k < j and '*/' in lines[k - 1]:
                    k = 0                         # another comment's end: not one block
                    break
                k -= 1
            if k < 1:
                break
            j = k - 1
        elif s.endswith(']') and (k := attr_start(lines, j)) is not None:
            j = k - 1
        else:
            break
    return j + 1


def ends_of(fns, lines):
    """Last line of each fn's span: the line before the next fn's leading
    attributes and comments, or the end of the file for the last fn. Lines
    between two fns therefore belong to neither, so adding `#[allow(..)]` or a
    doc comment above a fn does not touch the fn before it."""
    ends = []
    for i, (start, _, _) in enumerate(fns):
        if i + 1 < len(fns):
            nxt = fns[i + 1][0]
            ends.append(max(start, lead_start(lines, nxt) - 1))
        else:
            ends.append(max(start, len(lines)))
    return ends


def owner(fns, ends, line):
    """Index of the fn whose span holds `line`, or None when no fn's does
    (before the first fn, or in the attributes and comments above a fn)."""
    best = None
    for i, (start, _, _) in enumerate(fns):
        if start <= line:
            best = i
        else:
            break
    if best is None or line > ends[best]:
        return None
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
