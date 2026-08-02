#!/bin/bash
# Coverage of the graph crate by the benchmark query set.
# Builds an instrumented debug module, runs every query once, reports
# line coverage of graph/src (excluding the generated GraphBLAS.rs FFI file).
set -euo pipefail
cd "$(dirname "$0")/.."

# macOS builds a .dylib, Linux a .so.
EXT=$([ "$(uname -s)" = Darwin ] && echo dylib || echo so)

COVDIR=bench/results/cov
rm -rf "$COVDIR" && mkdir -p "$COVDIR"

# The link-arg is required on Linux (and so in CI): the embedded RediSearch
# static libs otherwise fail to link with duplicate-symbol errors. macOS's
# linker does not take the flag and does not need it. Same pair the coverage
# skill uses for the full-suite run.
COV_RUSTFLAGS="-C instrument-coverage"
if [ "$(uname -s)" != Darwin ]; then
  COV_RUSTFLAGS="$COV_RUSTFLAGS -C link-arg=-Wl,--allow-multiple-definition"
  # graph/build.rs compiles C++ shims; the toolchain image has no default CXX.
  export CXX="${CXX:-clang++}"
fi

echo "== instrumented debug build =="
RUSTFLAGS="$COV_RUSTFLAGS" cargo build

echo "== running query set once each =="
LLVM_PROFILE_FILE="$PWD/$COVDIR/cov-%p.profraw" \
  python3 bench/run_bench.py --once --port 6401 \
  --module "$PWD/target/debug/libfalkordb.$EXT"

TOOLS=$(dirname "$(find ~/.rustup/toolchains -name llvm-profdata | head -1)")
# llvm-profdata ships in the llvm-tools component, which is not installed by
# default. Without this check TOOLS becomes "." and the next line fails with a
# confusing "./llvm-profdata: No such file or directory".
if [ ! -x "$TOOLS/llvm-profdata" ]; then
  echo "llvm-profdata not found. Install it with: rustup component add llvm-tools-preview" >&2
  exit 1
fi
"$TOOLS/llvm-profdata" merge --sparse "$COVDIR"/*.profraw -o "$COVDIR/cov.profdata"

"$TOOLS/llvm-cov" report --instr-profile "$COVDIR/cov.profdata" \
  "target/debug/libfalkordb.$EXT" \
  --ignore-filename-regex='(GraphBLAS\.rs|\.cargo|rustc)' \
  > "$COVDIR/report.txt"

echo "== graph crate coverage (excluding GraphBLAS.rs FFI) =="
awk '/graph\/src/ {tot += $8; miss += $9}
     END {printf "lines: %d/%d = %.1f%%\n", tot - miss, tot, (tot - miss) / tot * 100}' \
  "$COVDIR/report.txt"

echo
echo "== least-covered graph/src files (>200 lines) =="
awk '/graph\/src/ && $8 > 200 {printf "%7.1f%%  %6d lines  %s\n", ($8 - $9) / $8 * 100, $8, $1}' \
  "$COVDIR/report.txt" | sort -n | head -15

echo
echo "full report: $COVDIR/report.txt"
