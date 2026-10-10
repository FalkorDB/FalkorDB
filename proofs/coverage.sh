#!/usr/bin/env bash
# Aggregate Lean-verification coverage.
#
# Concatenates every proofs/*/COVERAGE.tsv (columns: file, function, line,
# bucket, lean_name_or_reason; header row optional, '#' lines are comments),
# cross-checks each row against the Rust functions found by `grep -n "fn "` over
# graph/src and src, and prints per-file and total counts / % by bucket.
#
# Rust source root: $COVERAGE_RUST_ROOT if set, else this checkout. The proofs
# cite origin/main; when this checkout is on another branch, point
# COVERAGE_RUST_ROOT at a worktree of origin/main (`git worktree add --detach
# <dir> origin/main`).
#
# Buckets: PROVEN, AXIOMATISED, MODELLED, NOT COVERED. A function listed by
# several projects counts once, in its best bucket (PROVEN > AXIOMATISED >
# MODELLED > NOT COVERED). Rust functions no TSV mentions are UNLISTED.
#
# Test code (wave 5 change): a function is TEST if it is
#   - inside a `#[cfg(test)]` module (inline, or an out-of-line `mod x;` file),
#   - a single `#[cfg(test)]` fn,
#   - in a file named *_bench.rs, tests.rs or test_aux.rs, or
#   - listed by some TSV row whose bucket is TEST-ONLY, or NOT COVERED with a
#     reason starting `TEST-ONLY:` (and by no row with a better bucket).
# TEST functions are excluded from every denominator (fns, cov%, prov%) and
# reported in the separate TEST column / totals. TEST-ONLY rows pointing at a
# function the scanner considers production code are listed as warnings.
# "non-FFI" totals additionally drop AXIOMATISED (true-FFI) fns from the
# denominator: prov% = PROVEN / (non-test fns - AXIOMATISED).
#
# Usage: proofs/coverage.sh [--all-files] [--unmatched] [--fns FILE]
#   --all-files   also print files with no coverage rows
#   --unmatched   list TSV rows that match no Rust fn (stale line/name/path)
#   --fns FILE    print "line<TAB>name<TAB>test|prod" for one Rust file and exit
set -euo pipefail
PROOFS="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$PROOFS/.." && pwd)"
if [[ -n "${COVERAGE_RUST_ROOT:-}" ]]; then
  RUST_ROOT="$COVERAGE_RUST_ROOT"
else
  RUST_ROOT="$REPO"
fi
exec python3 - "$PROOFS" "$RUST_ROOT" "$@" <<'PY'
import glob, os, re, sys
from collections import defaultdict

proofs, repo, *flags = sys.argv[1:]
ALL_FILES = '--all-files' in flags
UNMATCHED = '--unmatched' in flags
FNS_FILE = flags[flags.index('--fns') + 1] if '--fns' in flags else None
BUCKETS = ['PROVEN', 'AXIOMATISED', 'MODELLED', 'NOT COVERED']
RANK = {b: i for i, b in enumerate(BUCKETS)}
TEST = 'TEST-ONLY'
ALIASES = {'AXIOMATIZED': 'AXIOMATISED', 'MODELED': 'MODELLED', 'NOT_COVERED': 'NOT COVERED',
           'NOTCOVERED': 'NOT COVERED', 'UNCOVERED': 'NOT COVERED', 'TEST_ONLY': TEST,
           'TESTONLY': TEST}
FN_RE = re.compile(r'\bfn\s+([A-Za-z_][A-Za-z0-9_]*)')
MOD_BLOCK = re.compile(r'(pub(\([^)]*\))?\s+)?mod\s+\w+\s*\{')
MOD_DECL = re.compile(r'(pub(\([^)]*\))?\s+)?mod\s+(\w+)\s*;')
TEST_FILES = re.compile(r'(_bench|^tests|^test_aux)\.rs$')

# ---- 1. Rust functions (grep -n "fn "), each tagged test / prod ----
rust = {}            # relpath -> {line: name}
test_fns = set()     # (relpath, line) of test functions
test_mod_files = set()
files = []
for top in ('graph/src', 'src'):
    for dp, dns, fns in os.walk(os.path.join(repo, top)):
        dns[:] = sorted(d for d in dns if d not in ('tests', 'target'))
        files += [os.path.join(dp, f) for f in sorted(fns) if f.endswith('.rs')]

def scan(p):
    rel = os.path.relpath(p, repo)
    lines = open(p, encoding='utf-8', errors='replace').read().split('\n')
    found, tests = {}, set()
    in_test, depth, pending_cfg_test = False, 0, False
    whole = bool(TEST_FILES.search(os.path.basename(p)))
    for k, line in enumerate(lines, 1):
        s = line.strip()
        code = line.split('//')[0]
        m = None if s.startswith('//') else FN_RE.search(code)
        if m and 'fn(' in code[:m.start() + 3]:
            m = None
        if in_test:
            if m:
                found[k] = m.group(1); tests.add(k)
            depth += line.count('{') - line.count('}')
            if depth <= 0:
                in_test = False
            continue
        if s.startswith('#[cfg(test)]'):
            pending_cfg_test = True
            continue
        if pending_cfg_test:
            if s.startswith('#['):
                continue
            pending_cfg_test = False
            if MOD_BLOCK.match(s):
                in_test, depth = True, line.count('{') - line.count('}')
                if depth <= 0:
                    in_test = False
                continue
            md = MOD_DECL.match(s)
            if md:                      # out-of-line test module file
                d = os.path.dirname(p)
                if os.path.basename(p) not in ('mod.rs', 'lib.rs', 'main.rs'):
                    d = os.path.join(d, os.path.basename(p)[:-3])
                for c in (os.path.join(d, md.group(3) + '.rs'), os.path.join(d, md.group(3), 'mod.rs')):
                    test_mod_files.add(os.path.relpath(c, repo))
                continue
            if m:                       # a single #[cfg(test)] fn
                found[k] = m.group(1); tests.add(k)
                continue
        if m:
            found[k] = m.group(1)
            if whole:
                tests.add(k)
    return rel, found, tests

for p in files:
    rel, found, tests = scan(p)
    if found:
        rust[rel] = found
        test_fns.update((rel, k) for k in tests)
for rel in test_mod_files:          # out-of-line #[cfg(test)] module files
    for k in rust.get(rel, {}):
        test_fns.add((rel, k))

if FNS_FILE:
    rel = os.path.relpath(os.path.join(repo, FNS_FILE), repo)
    for ln, nm in sorted(rust.get(rel, {}).items()):
        print(f"{ln}\t{nm}\t{'test' if (rel, ln) in test_fns else 'prod'}")
    sys.exit(0)

by_base = defaultdict(list)
for rel in rust:
    by_base[os.path.basename(rel)].append(rel)

def resolve(fpath, name, line):
    """Map a TSV (file, function, line) to (relpath, line) of a real Rust fn."""
    f = fpath.strip().lstrip('./')
    cands = [r for r in rust if r == f or r.endswith('/' + f)] or by_base.get(os.path.basename(f), [])
    short = name.strip().split('/')[0].split('::')[-1].split('.')[-1].split('(')[0].strip()
    # exact line + name, then name within +-10 lines, then unique name in file
    for r in cands:
        if rust[r].get(line) == short:
            return r, line
    best = None
    for r in cands:
        for ln, nm in rust[r].items():
            if nm == short and abs(ln - line) <= 10 and (best is None or abs(ln - line) < abs(best[1] - line)):
                best = (r, ln)
    if best:
        return best
    hits = [(r, ln) for r in cands for ln, nm in rust[r].items() if nm == short]
    return hits[0] if len(hits) == 1 else None

# ---- 2. COVERAGE.tsv rows ----
tsvs = sorted(glob.glob(os.path.join(proofs, '*', 'COVERAGE.tsv')))
projects = sorted(d for d in os.listdir(proofs)
                  if os.path.isfile(os.path.join(proofs, d, 'lakefile.toml'))
                  or os.path.isfile(os.path.join(proofs, d, 'lakefile.lean')))
have = {os.path.basename(os.path.dirname(t)) for t in tsvs}
missing = [p for p in projects if p not in have]

best = {}                       # (rel, line) -> bucket (BUCKETS or TEST)
test_rows = {}                  # (rel, line) -> tsv location, for TEST-ONLY rows
unmatched, bad_bucket, nrows = [], [], 0
for t in tsvs:
    proj = os.path.basename(os.path.dirname(t))
    for k, raw in enumerate(open(t, encoding='utf-8', errors='replace'), 1):
        cols = raw.rstrip('\n').split('\t')
        if len(cols) < 4 or cols[0].strip().lower() == 'file' or not cols[0].strip() or raw.startswith('#'):
            continue
        nrows += 1
        b = cols[3].strip().upper()
        b = ALIASES.get(b.replace(' ', '_'), ALIASES.get(b, b))
        reason = cols[4].strip() if len(cols) > 4 else ''
        if b == 'NOT COVERED' and reason.upper().startswith('TEST-ONLY'):
            b = TEST
        if b not in RANK and b != TEST:
            bad_bucket.append(f'{proj}/COVERAGE.tsv:{k}: bucket {cols[3]!r}')
            continue
        try:
            line = int(cols[2].strip().split('-')[0] or 0)
        except ValueError:
            line = 0
        hit = resolve(cols[0], cols[1], line)
        if not hit:
            unmatched.append(f'{proj}/COVERAGE.tsv:{k}: {cols[0]}\t{cols[1]}\t{cols[2]}')
            continue
        if b == TEST:
            test_rows.setdefault(hit, f'{proj}/COVERAGE.tsv:{k}')
            best.setdefault(hit, TEST)
        elif hit not in best or best[hit] == TEST or RANK[b] < RANK[best[hit]]:
            best[hit] = b

# a fn is TEST if the scanner says so, or its best row is TEST-ONLY
suspicious = [f'{loc}: {r}:{ln} {rust[r][ln]}' for (r, ln), loc in sorted(test_rows.items())
              if (r, ln) not in test_fns]
for hit, b in best.items():
    if b == TEST:
        test_fns.add(hit)

# ---- 3. report ----
per = defaultdict(lambda: defaultdict(int))
for rel in rust:
    for ln in rust[rel]:
        key = (rel, ln)
        if key in test_fns:
            per[rel]['TEST'] += 1
            if key in best:
                per[rel]['_listed'] += 1
        elif key in best:
            per[rel][best[key]] += 1
            per[rel]['_listed'] += 1
hdr = (f"{'file':58} {'fns':>5} {'PROV':>5} {'AXIOM':>5} {'MODEL':>5} {'NOTCV':>5} {'UNLST':>5} "
       f"{'cov%':>6} {'prov%':>6} {'TEST':>5}")
print(f'Rust root: {repo}')
print(hdr); print('-' * len(hdr))
def row(label, n, c):
    listed = sum(c[b] for b in BUCKETS)
    cov = c['PROVEN'] + c['AXIOMATISED'] + c['MODELLED']
    pct = lambda x: f'{100.0 * x / n:5.1f}%' if n else '    -'
    return (f"{label[:58]:58} {n:5d} {c['PROVEN']:5d} {c['AXIOMATISED']:5d} {c['MODELLED']:5d} "
            f"{c['NOT COVERED']:5d} {n - listed:5d} {pct(cov):>6} {pct(c['PROVEN']):>6} {c['TEST']:5d}")
tot, tc = defaultdict(int), defaultdict(int)
tot_n = tf = touched = 0
for rel in sorted(rust):
    c = defaultdict(int, per.get(rel, {}))
    n = len(rust[rel]) - c['TEST']          # non-test fns
    listed = c['_listed'] > 0
    tot_n += n
    for b in BUCKETS + ['TEST']:
        tot[b] += c[b]
    if listed:
        touched += 1; tf += n
        for b in BUCKETS + ['TEST']:
            tc[b] += c[b]
    if listed or ALL_FILES:
        print(row(rel, n, c))
print('-' * len(hdr))
print(row(f'TOTAL over {touched} targeted files', tf, tc))
print(row(f'TOTAL over all {len(rust)} Rust files (graph/src + src)', tot_n, tot))
all_fns = sum(len(v) for v in rust.values())
nonffi = tot_n - tot['AXIOMATISED']
print()
print(f'all fns {all_fns} = non-test {tot_n} + TEST {tot["TEST"]} (excluded from denominators)')
print(f'non-test PROVEN {tot["PROVEN"]}/{tot_n} = {100.0 * tot["PROVEN"] / max(tot_n, 1):.1f}%; '
      f'non-FFI non-test PROVEN {tot["PROVEN"]}/{nonffi} = {100.0 * tot["PROVEN"] / max(nonffi, 1):.1f}% '
      f'(AXIOMATISED {tot["AXIOMATISED"]} dropped)')
print(f'{len(tsvs)} COVERAGE.tsv files, {nrows} rows, {len(best)} distinct fns matched, '
      f'{len(unmatched)} rows unmatched, {len(bad_bucket)} bad buckets')
if missing:
    print(f'projects without COVERAGE.tsv ({len(missing)}): {" ".join(missing)}')
for b in bad_bucket:
    print('  bad bucket:', b)
if suspicious:
    print(f'TEST-ONLY rows on fns the scanner sees as production code ({len(suspicious)}):')
    for s in suspicious:
        print('  test-only?:', s)
if UNMATCHED:
    for u in unmatched:
        print('  unmatched:', u)
elif unmatched:
    print('  (run with --unmatched to list them)')
PY
