#!/bin/bash
# H2 repro: `book` is correct only together with `pk` and `hb2` (`npar` by reading the code; no run here).
#
# Builds the m1-optimize bench from its patch (README of ../../prototypes/code/m1-optimize) in a
# temporary worktree under $H, downloads SNAP roadNet-PA and checks its sha256, then runs the matrix
# below on PA, depths 0-3, 1 thread, check run only, against the saved baseline hashes
# ../../prototypes/code/m1-optimize/results/hashes/pa8_base.hash.
#
# Usage: run.sh [setup|build|runs|trim|clean|all]   (default: all = setup build runs trim)
# Needs: network (SNAP, submodules), Homebrew LLVM for CC/CXX on macOS, ~10 GB free for the target dir.
set -euo pipefail

REPO=${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
H=${H:-$(dirname "$REPO")/h2-tmp}          # temporary root, beside the checkout, outside /tmp
HERE="$REPO/.handover/cch-redesign/repro/h2-switch-dependency"
PROTO="$REPO/.handover/cch-redesign/prototypes/code/m1-optimize"
BASE=7df8e588727f47619d9567d111c7fc26a2208caf
HASHES="$PROTO/results/hashes/pa8_base.hash"
PA_GZ_SHA=450b8733635d887466a2b96b26411f6e62caf7006f8a264f59cf9b5d75cdf549
PA_TXT_SHA=e7e9abdd06b512e47a6e63d056c1d1e0ad2cda14f993bf77c6217526f00579b1
FINAL=hb2,hsplit,cpack,hcon,pk,book,hkey,hpar,gath,usel,zmatch,npar

setup() {
  mkdir -p "$H/data" "$H/cache" "$H/logs"
  git -C "$REPO" worktree add --detach "$H/wt" "$BASE"
  git -C "$H/wt" apply "$PROTO/m1opt.patch"
  git -C "$H/wt" submodule update --init --recursive
  (cd "$H/data" && curl -sS -O https://snap.stanford.edu/data/roadNet-PA.txt.gz)
  echo "$PA_GZ_SHA  $H/data/roadNet-PA.txt.gz" | shasum -a 256 -c -
  gunzip -kf "$H/data/roadNet-PA.txt.gz"
  echo "$PA_TXT_SHA  $H/data/roadNet-PA.txt" | shasum -a 256 -c -
}

build() {
  cd "$H/wt"
  # macOS needs Homebrew's clang for OpenMP (see the repo README); on Linux keep the system compilers.
  if [ "$(uname)" = Darwin ]; then export CC=${CC:-/opt/homebrew/opt/llvm/bin/clang} CXX=${CXX:-/opt/homebrew/opt/llvm/bin/clang++}; fi
  export CARGO_TARGET_DIR="$H/target" CARGO_PROFILE_RELEASE_DEBUG=line-tables-only
  cargo test --release -p graph --features index-falkordb --lib --no-run 2>&1 | tee "$H/logs/build.log" | tail -3
  BIN=$(grep -o "$H/target/release/deps/graph-[0-9a-f]*" "$H/logs/build.log" | tail -1)
  cp "$BIN" "$H/graph-test"
  "$H/graph-test" --list --ignored 2>/dev/null | grep m1_reuse
}

# one run: NAME M1O
one() {
  local name=$1 opts=$2 log="$H/logs/$1.log" rc=0
  { echo "BIN md5=$(md5 -q "$H/graph-test")"; echo "ENV M1O=$opts M1_THREADS=1 M1_RUNS=0 M1_DEPTHS=3 M1_FERAL=0"; } > "$log"
  env M1_JIT=run M1_CT=1 M1_GRAPH="$H/data/roadNet-PA.txt" M1_THREADS=1 M1_RUNS=0 M1_DEPTHS=3 \
      M1_FERAL=0 M1_PERM_CACHE="$H/cache/PA.perm" M1_HASH="check:$HASHES" M1O="$opts" \
      "$H/graph-test" --ignored --exact graph::graphblas::m1_coarsen_bench::reuse::m1_reuse --nocapture \
      >> "$log" 2>&1 || rc=$?
  local cmp fail mism check
  cmp=$(grep -o 'hash_compared: [0-9]*' "$log" | tail -1 | grep -o '[0-9]*' || echo 0)
  fail=$(grep -o 'hash_fail: [0-9]*' "$log" | tail -1 | grep -o '[0-9]*' || echo 0)
  mism=$(grep -c '^HASHMISMATCH' "$log" || true)
  check=$(grep -m1 'CHECK FAILED' "$log" | sed 's/.*CHECK FAILED/CHECK FAILED/' || true)
  printf '%-12s | %-52s | exit %-3s | compared %-2s | failed %-2s | %s\n' "$name" "$opts" "$rc" "$cmp" "$fail" "${check:-}"
}

runs() {
  # the feral order is written on the first run (M1_FERAL=0 with no cache runs feral once)
  {
    one final "$FINAL"
    one hb2_pk_book "hb2,pk,book"
    one book "book"
    one hb2_book "hb2,book"
    one pk_book "pk,book"
  } | tee "$H/logs/summary.txt"
}

# keep the lines that carry the result: run header, per-path hashes, mismatches, checks, the panic
trim() {
  mkdir -p "$HERE/results"
  cp "$H/logs/summary.txt" "$HERE/results/summary.txt"
  for f in "$H"/logs/*.log; do
    case "$(basename "$f")" in build*) continue ;; esac
    { grep -E '^(BIN|ENV|HASH|LABELHASH|CHECKS|CSRTYPES|JIT)|HASHMISMATCH|HASHDIFF|CHECK FAILED|panicked|^test |test result|hash_compared' "$f" || true; } \
      | cut -c1-400 > "$HERE/results/$(basename "$f")"
  done
  grep -E '^ *(Finished|Executable)|^error' "$H/logs/build.log" > "$HERE/results/build.txt" || true
  du -sh "$HERE/results"
}

clean() {
  git -C "$REPO" worktree remove --force "$H/wt" || true
  rm -rf "$H"
}

case "${1:-all}" in
  setup) setup ;; build) build ;; runs) runs ;; trim) trim ;; clean) clean ;;
  all) setup; build; runs; trim ;;
  *) echo "usage: $0 [setup|build|runs|trim|clean|all]"; exit 2 ;;
esac
