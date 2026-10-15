#!/bin/sh
#
# Regenerate the vendored single-file zstd amalgamation.
#
# Usage:  ./regenerate.sh /path/to/zstd-1.5.7
#
# Everything this produces is checked in, so you only need to run it to bump
# the zstd version. See PROVENANCE.md for why each step is here. The final
# step is a CHECK, not a formality: it is what catches a symbol escaping the
# rename, and one already has (ZSTD_isError - see step 4).
#
set -eu

SRC="${1:?usage: regenerate.sh /path/to/zstd-x.y.z}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PREFIX=FDB_

test -f "$SRC/lib/zstd.h" || { echo "not a zstd source tree: $SRC" >&2; exit 1; }
printf 'zstd source: %s (' "$SRC"
awk '/define ZSTD_VERSION_(MAJOR|MINOR|RELEASE)/ {printf "%s", $3"."}' "$SRC/lib/zstd.h"
printf ')\n'

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# ---------------------------------------------------------------------------
# 1. our amalgamation template: upstream's, minus two things we never call
#
#    - ZSTD_MULTITHREAD: a worker thread pool inside a Redis module, for a
#      synchronous level-1 compress on the write path. zstd only spawns
#      workers when nbWorkers > 0, so this is dead code rather than a
#      behaviour change, and single-threaded is upstream's own default.
#    - dictBuilder/*: we never train dictionaries. divsufsort alone is large.
#
#    Measured, compiled -O2: 493,800 vs 595,608 bytes of object and 333 vs
#    371 global symbols. Both are supported upstream configurations.
# ---------------------------------------------------------------------------
sed -e 's|^#ifndef __EMSCRIPTEN__$|#if 0 /* FalkorDB: no MT compression on the write path */|' \
    -e '/^#include "dictBuilder\//d' \
    "$SRC/build/single_file_libs/zstd-in.c" > "$WORK/zstd-fdb-in.c"

# ---------------------------------------------------------------------------
# 2. amalgamate
# ---------------------------------------------------------------------------
( cd "$SRC/build/single_file_libs" \
  && python3 ./combine.py -r ../../lib -x legacy/zstd_legacy.h \
       -o "$WORK/zstd-raw.c" "$WORK/zstd-fdb-in.c" ) >/dev/null
test -s "$WORK/zstd-raw.c"

# ---------------------------------------------------------------------------
# 3. discover the symbols to rename, by compiling and asking nm.
#    NOT a hand-maintained list: see PROVENANCE.md.
# ---------------------------------------------------------------------------
cc -O2 -std=gnu11 -c "$WORK/zstd-raw.c" -o "$WORK/probe.o"
nm -g "$WORK/probe.o" | grep -E '^[0-9a-f]+ [TDSBC] ' | awk '{print $3}' \
  | sed 's/^_//' | sort -u > "$WORK/syms.txt"
echo "symbols to rename: $(wc -l < "$WORK/syms.txt" | tr -d ' ')"

# ---------------------------------------------------------------------------
# 4. emit the amalgamation with the rename applied.
#
#    Two edits, and only two:
#
#    a. include zstd_symbols.h first, so every declaration and definition in
#       the file is renamed.
#
#    b. RE-APPLY the rename after any `#undef` of a symbol we renamed.
#       zstd_internal.h macros ZSTD_isError to ERR_isError "for inlining",
#       and zstd_common.c then does `#undef ZSTD_isError` before defining the
#       real exported function. That #undef removes OUR macro too, so without
#       this the definition emits a bare `ZSTD_isError` - which is precisely
#       the symbol that collides with GraphBLAS. Step 6 catches it if this
#       stops working.
# ---------------------------------------------------------------------------
{
  echo '/* FalkorDB: added to the generated amalgamation - see PROVENANCE.md */'
  echo '#include "zstd_symbols.h"'
  awk -v prefix="$PREFIX" '
    NR == FNR { renamed[$0] = 1; next }
    {
      print
      if ($0 ~ /^[[:space:]]*#[[:space:]]*undef[[:space:]]/) {
        name = $0
        sub(/^[[:space:]]*#[[:space:]]*undef[[:space:]]+/, "", name)
        sub(/[[:space:]].*$/, "", name)
        if (name in renamed) {
          printf "#define %s %s%s  /* FalkorDB: re-applied after zstd'\''s own undef */\n", \
                 name, prefix, name
        }
      }
    }
  ' "$WORK/syms.txt" "$WORK/zstd-raw.c"
} > "$HERE/zstd.c"

# ---------------------------------------------------------------------------
# 5. emit the rename header and copy the pristine public headers
# ---------------------------------------------------------------------------
cp "$SRC/lib/zstd.h" "$SRC/lib/zstd_errors.h" "$HERE/"
awk -v prefix="$PREFIX" '{ printf "#define %-46s %s%s\n", $0, prefix, $0 }' \
  "$WORK/syms.txt" > "$WORK/defines.txt"
# preserve the hand-written preamble, replace only the generated defines
sed '/^#define /,$d' "$HERE/zstd_symbols.h" > "$WORK/preamble.txt" 2>/dev/null \
  || : > "$WORK/preamble.txt"
test -s "$WORK/preamble.txt" || { echo "zstd_symbols.h preamble missing" >&2; exit 1; }
cat "$WORK/preamble.txt" "$WORK/defines.txt" > "$HERE/zstd_symbols.h"

# ---------------------------------------------------------------------------
# 6. THE CHECK. Zero bare zstd globals may remain.
# ---------------------------------------------------------------------------
cc -O2 -std=gnu11 -c "$HERE/zstd.c" -o "$WORK/final.o" -I"$HERE"
LEAKED=$(nm -g "$WORK/final.o" | grep -E '^[0-9a-f]+ [TDSBC] ' | awk '{print $3}' \
         | sed 's/^_//' | grep -vE "^$PREFIX" || true)
if [ -n "$LEAKED" ]; then
  echo "FAIL: global symbols escaped the rename:" >&2
  echo "$LEAKED" >&2
  exit 1
fi
echo "OK: $(nm -g "$WORK/final.o" | grep -cE '^[0-9a-f]+ [TDSBC] ') globals, all ${PREFIX}-prefixed, 0 bare"
