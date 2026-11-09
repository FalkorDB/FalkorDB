#!/usr/bin/env python3
"""Lean-free checks of a re-prove patch, for proof_gate.sh (run from proofs-src/
with the patch staged).

Usage: proof_gate_checks.py <base sha> <impact.json> <verdicts.json> <added_theorems.json out>

Prints the Markdown report; its last line holds the tags REJECT (drop the patch),
HELD (a theorem statement, its context or a coverage row changed) and INCOMPLETE
(a changed function has no valid verdict).
"""
import json, re, subprocess, sys
base, impact_path, verdicts_path, added_path = sys.argv[1:]

def git(*a):
    return subprocess.run(['git', *a], capture_output=True, text=True).stdout

def strip_comments(src):
    """Blank -- and nested /- -/ comments, keeping string literals and newlines."""
    out, i, n, depth, in_str = [], 0, len(src), 0, False
    while i < n:
        c = src[i]
        if in_str:
            if c == '\\' and i + 1 < n:
                out.append(src[i:i + 2]); i += 2; continue
            if c == '"':
                in_str = False
            out.append(c); i += 1; continue
        if depth == 0 and c == '"':
            in_str = True; out.append(c); i += 1; continue
        if src.startswith('/-', i):
            depth += 1; out.append('  '); i += 2; continue
        if depth and src.startswith('-/', i):
            depth -= 1; out.append('  '); i += 2; continue
        if depth:
            out.append('\n' if c == '\n' else ' '); i += 1; continue
        if src.startswith('--', i):
            j = src.find('\n', i)
            j = n if j < 0 else j
            out.append(' ' * (j - i)); i = j; continue
        out.append(c); i += 1
    return ''.join(out)

HEAD_RE = re.compile(r'(?:theorem|lemma)\s+([^\s(:{\[]+)')
NS_RE = re.compile(r'^\s*(namespace|section|end)\b\s*([^\s]*)', re.M)

def statement(src, k):
    """Text from k to the first := outside brackets and strings."""
    depth, i, in_str = 0, k, False
    while i < len(src):
        c = src[i]
        if in_str:
            if c == '\\':
                i += 2; continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        elif depth == 0 and src.startswith(':=', i):
            break
        i += 1
    # Collapse whitespace outside string literals only.
    parts = re.split(r'("(?:\\.|[^"\\])*")', src[k:i])
    return ''.join(p if p.startswith('"') else ' '.join(p.split()) + ' ' for p in parts).strip()

# Commands that change what an unchanged statement means: section variables
# (`variable (h : False)` + `include h` gives a theorem an impossible
# premise), name resolution (`open`), typeclass resolution (`instance`,
# `attribute [instance]`) and notation.
CONTEXT_RE = re.compile(r'^[ \t]*(?:local[ \t]+|scoped[ \t]+)?(variable|include|omit|open|export|attribute|instance|notation|'
                        r'infix|infixl|infixr|prefix|postfix|macro|syntax|universe|set_option)\b', re.M)

def command_text(src, k):
    """A command from k: its line plus the indented lines that continue it."""
    lines = src[k:].split('\n')
    out = [lines[0]]
    for line in lines[1:]:
        if not line.strip() or line[0] not in ' \t':
            break
        out.append(line)
    return ' '.join(' '.join(out).split())

def decls(text):
    """Fully qualified name -> (statement, the context commands before it, in order).

    A name seen twice maps to None. The context is every namespace/section/end
    and context command above the declaration, so adding, editing or moving
    one changes it.
    """
    src = strip_comments(text)
    scopes, events, ctx = [], [], []
    for m in NS_RE.finditer(src):
        events.append((m.start(), m.group(1), m.group(2)))
    for m in CONTEXT_RE.finditer(src):
        events.append((m.start(), 'context', command_text(src, m.start())))
    for m in HEAD_RE.finditer(src):
        if m.start() and src[m.start() - 1] not in ' \t\n]':
            continue
        events.append((m.start(), 'decl', m))
    found = {}
    for pos, kind, arg in sorted(events, key=lambda e: e[0]):
        if kind in ('namespace', 'section'):
            scopes.append((kind, arg)); ctx.append(f'{kind} {arg}')
        elif kind == 'end':
            if scopes:
                scopes.pop()
            ctx.append(f'end {arg}')
        elif kind == 'context':
            ctx.append(arg)
        else:
            ns = [name for k2, name in scopes if k2 == 'namespace' and name]
            full = '.'.join(ns + [arg.group(1)])
            found[full] = None if full in found else (statement(src, arg.end()), tuple(ctx))
    return found

problems = []
# Build-time code added to a .lean file.
# Code that runs while Lean elaborates, and unproven declarations. Matched on
# the added lines of each file after stripping that whole file's comments,
# so a word inside a multi-line doc comment does not count.
FORBID = re.compile(r'(#eval|#exec|#exit|\brun_cmd\b|\brun_elab\b|\brun_meta\b|\brun_tac\b|'
                    r'\binitialize\b|\bbuiltin_initialize\b|\belab\b|\belab_rules\b|'
                    r'\bmacro_rules\b|\bIO\b|\bunsafe\b|\bimplemented_by\b|\bextern\b|'
                    r'\baxiom\b|\bsorry\b|\badmit\b|\bsorryAx\b|AXIOM-OK)')
added = {}
cur = None
for line in git('diff', '--cached', '--no-renames', '-U0', '--', '*.lean').splitlines():
    if line.startswith('+++ '):
        cur = line[6:] if line.startswith('+++ b/') else None
    elif line.startswith('@@') and cur:
        m = re.match(r'@@ -\S+ \+(\d+)(?:,(\d+))? @@', line)
        start, count = int(m.group(1)), int(m.group(2) or 1)
        added.setdefault(cur, set()).update(range(start, start + count))
for f, lines in added.items():
    text = strip_comments(git('show', f':{f}')).split('\n')
    for n in sorted(lines):
        if n - 1 < len(text):
            m = FORBID.search(text[n - 1])
            if m:
                problems.append(f'- `{f}:{n}` adds `{m.group(1)}`: `{text[n - 1].strip()[:120]}`')
if problems:
    print('Patch dropped: it adds build-time code or an unproven declaration:')
    print('\n'.join(problems))
    print('REJECT')
    sys.exit(0)

held = []
# Existing theorems: the statement and the context above it must be unchanged.
files = git('diff', '--cached', '--name-only', '--no-renames', '--diff-filter=MD', '--', '*.lean').split()
for f in files:
    old, new = decls(git('show', f'{base}:{f}')), decls(git('show', f':{f}'))
    for name, st in old.items():
        if name not in new:
            held.append(f'- `{f}` `{name}`: removed')
        elif st is None or new[name] is None:
            held.append(f'- `{f}` `{name}`: name declared more than once; review by hand')
        elif new[name][0] != st[0]:
            held.append(f'- `{f}` `{name}`: statement changed')
        elif new[name][1] != st[1]:
            came = [c for c in new[name][1] if c not in st[1]]
            gone = [c for c in st[1] if c not in new[name][1]]
            what = '; '.join([f'adds `{c[:80]}`' for c in came] + [f'removes `{c[:80]}`' for c in gone]) or 'reordered'
            held.append(f'- `{f}` `{name}`: context above it changed ({what})')

# The theorems the patch adds: fully qualified names of declarations in the
# patched .lean files (comments stripped, so neither a comment nor a Markdown
# file counts). lean_ci.sh builds them below; the report job posts a
# counterexample review only for a theorem named here.
added_thms = []
for f in git('diff', '--cached', '--name-only', '--no-renames', '--diff-filter=AM', '--', '*.lean').split():
    old, new = decls(git('show', f'{base}:{f}')), decls(git('show', f':{f}'))
    added_thms += [n for n, st in new.items() if st is not None and n not in old]
json.dump(sorted(set(added_thms)), open(added_path, 'w'))

# COVERAGE.tsv: no existing function may lose its row or move to a worse bucket
# (a function counts no better than its worst row).
RANK = {b: i for i, b in enumerate(['PROVEN', 'CONDITIONAL', 'AXIOMATISED', 'MODELLED', 'NOT COVERED', 'TEST-ONLY'])}

def worst(text):
    out = {}
    for line in text.splitlines()[1:]:
        cols = line.split('\t')
        if len(cols) < 4:
            continue
        key, rank = (cols[0], cols[1]), RANK.get(cols[3].strip(), len(RANK))
        out[key] = max(out.get(key, rank), rank)
    return out

impact = json.load(open(impact_path))
gone_ok = {(r.get('file'), r.get('fn')) for r in impact.get('removed', []) if isinstance(r, dict)}

def bare(key):
    # pr_impact.load_rows()'s name: `Lexer::set_pos` -> `set_pos`, `f/Arm` -> `f`.
    name = key[1].strip().split('/')[0].split('::')[-1].split('.')[-1].split('(')[0].strip()
    return (key[0].strip().lstrip('./'), name)

for f in git('diff', '--cached', '--name-only', '--no-renames', '--diff-filter=MD', '--', 'proofs/*/COVERAGE.tsv').split():
    old, new = worst(git('show', f'{base}:{f}')), worst(git('show', f':{f}'))
    names = {v: k for k, v in RANK.items()}
    for key, rank in old.items():
        if key not in new and bare(key) not in gone_ok:
            held.append(f'- `{f}` `{key[0]}` `{key[1]}`: coverage row removed')
        elif key in new and new[key] > rank:
            held.append(f'- `{f}` `{key[0]}` `{key[1]}`: coverage {names.get(rank, "?")} → {names.get(new[key], "unknown bucket")}')
if held:
    print('Held for review: an existing theorem statement, the context above one, or a coverage row changed:')
    print('\n'.join(held))

VERDICTS = {'unchanged', 'model updated', 'counterexample'}
# A counterexample is posted as a review only with all of these (as in the report job).
CEX_FIELDS = ('theorem', 'why', 'repro', 'before', 'after')
try:
    records = [v for v in json.load(open(verdicts_path)) if isinstance(v, dict)]
    keys = [(str(v['file']), str(v['fn']), int(v['line'])) for v in records]
    given_full = dict(zip(keys, records))
    dups = sorted({k for k in keys if keys.count(k) > 1})
except Exception:
    given_full, dups = {}, []
given = {k: v.get('verdict') if isinstance(v.get('verdict'), str) else None for k, v in given_full.items()}
missing = [c for c in impact['changed']
           if given.get((c['file'], c['fn'], int(c['line']))) not in VERDICTS
           or (c['file'], c['fn'], int(c['line'])) in dups]
noname = [k for k, v in given.items() if v in VERDICTS and not given_full[k].get('theorems')]
nocex = [k for k, v in given.items() if v == 'counterexample'
         and not all(isinstance(given_full[k].get(f), str) and given_full[k].get(f).strip()
                     for f in CEX_FIELDS)]
if nocex:
    print('Counterexamples missing theorem, why, repro, before or after, so not posted as a review (reported, not gated):')
    print('\n'.join(f'- `{f}` `{fn}` (line {ln})' for f, fn, ln in nocex))
if noname:
    print('Verdicts that name no Lean theorem (reported, not gated):')
    print('\n'.join(f'- `{f}` `{fn}` (line {ln})' for f, fn, ln in noname))
if missing:
    print('No valid verdict for:')
    print('\n'.join(f"- `{c['file']}` `{c['fn']}` (line {c['line']})" for c in missing))
print('HELD' if held else '', 'INCOMPLETE' if missing else '')
