#!/usr/bin/env bash
# Lean proof CI for proofs/*/ .
#
# For every top-level Lake project proofs/<dir>/ (has lakefile.toml|lakefile.lean):
#   1. `lake build` must succeed;
#   2. its own .lean sources (excluding .lake/ and nested Lake projects, e.g.
#      aeneas_pilot/full/) must contain no `sorry`, no `admit`, no
#      `native_decide`, and no `axiom` unless that same line carries a
#      `-- AXIOM-OK: <justification>` comment.
#      Comments and docstrings are stripped before the scan,
#      so prose like "zero sorry" is fine.
#
# Usage: proofs/lean_ci.sh [--no-build] [dir ...]     (default: all projects)
# Exit status: 0 iff every project builds and is clean.
set -uo pipefail
PROOFS="$(cd "$(dirname "$0")" && pwd)"
BUILD=1
if [[ "${1:-}" == "--no-build" ]]; then BUILD=0; shift; fi

if [[ $# -gt 0 ]]; then
  DIRS=("$@")
else
  DIRS=()
  for d in "$PROOFS"/*/; do
    d="${d%/}"
    [[ -f "$d/lakefile.toml" || -f "$d/lakefile.lean" ]] && DIRS+=("$(basename "$d")")
  done
fi

scan() { # $1 = project dir; prints offending "file:line: kind: text", exit 1 if any
  python3 - "$1" <<'PY'
import os, re, sys
root = sys.argv[1]
bad = 0

def strip_comments(src):
    """Blank out -- line comments and nested /- -/ block comments (incl. docstrings),
    keeping newlines so line numbers survive. String literals are respected."""
    out, i, n, depth = [], 0, len(src), 0
    in_str = False
    while i < n:
        c = src[i]
        if depth == 0 and not in_str and c == '"':
            in_str = True; out.append(c); i += 1; continue
        if in_str:
            if c == '\\' and i + 1 < n:
                out.append('  '); i += 2; continue
            if c == '"': in_str = False
            out.append(c if c in '"\n' else ' '); i += 1; continue
        if src.startswith('/-', i):
            depth += 1; out.append('  '); i += 2; continue
        if depth > 0 and src.startswith('-/', i):
            depth -= 1; out.append('  '); i += 2; continue
        if depth > 0:
            out.append('\n' if c == '\n' else ' '); i += 1; continue
        if src.startswith('--', i):
            j = src.find('\n', i)
            j = n if j < 0 else j
            out.append(' ' * (j - i)); i = j; continue
        out.append(c); i += 1
    return ''.join(out)

nested = set()
for dp, dns, fns in os.walk(root):
    if dp != root and ('lakefile.toml' in fns or 'lakefile.lean' in fns):
        nested.add(dp); dns[:] = []
for dp, dns, fns in os.walk(root):
    dns[:] = [d for d in dns if d != '.lake' and os.path.join(dp, d) not in nested]
    for fn in sorted(fns):
        if not fn.endswith('.lean') or fn == 'lakefile.lean':
            continue
        p = os.path.join(dp, fn)
        src = open(p, encoding='utf-8', errors='replace').read()
        orig = src.split('\n')
        for k, line in enumerate(strip_comments(src).split('\n'), 1):
            rel = os.path.relpath(p, os.path.dirname(root))
            if re.search(r'(?<![\w.])(sorry|admit|native_decide)(?![\w.])', line):
                print(f"{rel}:{k}: sorry/admit/native_decide: {orig[k-1].strip()[:120]}"); bad += 1
            if re.search(r'(?<![\w.])axiom(?![\w.])', line) and '-- AXIOM-OK:' not in orig[k-1]:
                print(f"{rel}:{k}: axiom without '-- AXIOM-OK:': {orig[k-1].strip()[:120]}"); bad += 1
sys.exit(1 if bad else 0)
PY
}

FAILED=()
for d in "${DIRS[@]}"; do
  dir="$PROOFS/$d"
  status=ok; msgs=""
  if [[ $BUILD == 1 ]]; then
    log="$(mktemp)"
    if ! (cd "$dir" && lake build >"$log" 2>&1); then
      status="BUILD FAILED"
      msgs+="$(grep -E "^error" "$log" | head -5 | sed "s|^|    |")"$'\n'
    fi
    rm -f "$log"
  fi
  if ! out="$(scan "$dir")"; then
    [[ $status == ok ]] && status="UNCLEAN" || status="$status+UNCLEAN"
    msgs+="$(printf '%s\n' "$out" | head -20 | sed 's/^/    /')"$'\n'
    n="$(printf '%s\n' "$out" | wc -l | tr -d ' ')"
    [[ $n -gt 20 ]] && msgs+="    ... ($n findings)"$'\n'
  fi
  printf '%-22s %s\n' "$d" "$status"
  [[ -n $msgs ]] && printf '%s' "$msgs"
  [[ $status != ok ]] && FAILED+=("$d")
done

echo "----"
echo "${#DIRS[@]} projects, ${#FAILED[@]} failing${FAILED[*]:+: ${FAILED[*]}}"
[[ ${#FAILED[@]} -eq 0 ]]
