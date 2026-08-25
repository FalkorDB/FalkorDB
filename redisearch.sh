#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

# RediSearch lives at deps/RediSearch as a git submodule, the same layout the C
# engine uses on `master`. GIT owns that checkout: this script only BUILDS what
# is there, and never fetches, clones, resets or otherwise moves the pin. That
# is what keeps the pinned commit single-sourced in the gitlink
# (`160000 <sha>` in the tree) instead of also being spelled out here.
#
# So every consumer must populate the submodule before calling this:
#   locally   git submodule update --init --recursive
#   CI        actions/checkout with `submodules: recursive`
#   Docker    the image COPYs deps/RediSearch in (see build/Dockerfile); the
#             build context carries the checked-out submodule, which is also
#             what makes the COPY layer's cache key the submodule CONTENT --
#             it turns over exactly when the pin moves.
#
# New RediSearch work lands on falkordb/llapi-extensions-8.6. To move the pin:
#   git -C deps/RediSearch fetch origin && git -C deps/RediSearch checkout <sha>
#   git add deps/RediSearch
# and nothing else -- there is no second copy of the SHA to keep in step.
REDISEARCH_DIR="$ROOT/deps/RediSearch"

# Populated means "has files". An uninitialized submodule is an empty directory,
# which would otherwise fail much later inside RediSearch's own build with
# something unhelpful.
if [ -z "$(ls -A "$REDISEARCH_DIR" 2>/dev/null || true)" ]; then
  echo "ERROR: $REDISEARCH_DIR is empty -- the RediSearch submodule is not checked out." >&2
  echo "       Run:  git submodule update --init --recursive" >&2
  echo "       (in CI, set \`submodules: recursive\` on actions/checkout)" >&2
  exit 1
fi

# RediSearch's own nested submodules are build inputs, not optional extras --
# VectorSimilarity is the one that fails loudest, so name it. A non-recursive
# `git submodule update` leaves these empty.
if [ -z "$(ls -A "$REDISEARCH_DIR/deps/VectorSimilarity" 2>/dev/null || true)" ]; then
  echo "ERROR: $REDISEARCH_DIR/deps/VectorSimilarity is empty -- RediSearch's own" >&2
  echo "       submodules are not checked out. Run:" >&2
  echo "       git submodule update --init --recursive" >&2
  exit 1
fi

cd "$REDISEARCH_DIR"

# VecSim promotes some warnings to errors; relax for our toolchain. Restore the
# file afterwards so the submodule does not show up as dirty in the parent repo.
#
# Restore the BYTES we saw, not `git checkout --` of the tracked version: this
# script deliberately supports editing deps/RediSearch in place, and checking
# out would silently throw away uncommitted VecSim work.
VECSIM_CMAKE="deps/VectorSimilarity/src/VecSim/CMakeLists.txt"
VECSIM_BACKUP="$(mktemp)"
cp "$VECSIM_CMAKE" "$VECSIM_BACKUP"
restore_vecsim() {
  cp "$VECSIM_BACKUP" "$VECSIM_CMAKE" 2>/dev/null || true
  rm -f "$VECSIM_BACKUP"
}
trap restore_vecsim EXIT
if [[ "$(uname -s)" == "Darwin" ]]; then
  sed -i '' 's/-Werror//g' "$VECSIM_CMAKE"
else
  sed -i 's/-Werror//g' "$VECSIM_CMAKE"
fi

# Build RediSearch 8.6 as an embeddable STATIC library:
#   REDISEARCH_BUILD_AS_LIBRARY=ON  omit src/module_main.c (its RedisModule_OnLoad);
#                                   FalkorDB supplies its own OnLoad.
#   REDISEARCH_BUILD_SHARED=OFF     produce static libredisearch.a, not redisearch.so.
#                                   A standalone build defaults to SHARED, so this is required.
#   CMAKE_POSITION_INDEPENDENT_CODE=ON  the archive is linked into our cdylib.
# On macOS build.sh forces clang/clang++ from PATH (Homebrew LLVM, required for
# OpenMP in VectorSimilarity); on Linux pass the compiler via $CC/$CXX (the
# toolchain image uses versioned clang-NN).
EXTRA_CMAKE=""
[ -n "${CC:-}" ] && EXTRA_CMAKE="$EXTRA_CMAKE -DCMAKE_C_COMPILER=$CC"
[ -n "${CXX:-}" ] && EXTRA_CMAKE="$EXTRA_CMAKE -DCMAKE_CXX_COMPILER=$CXX"
export CMAKE_ARGS="-DREDISEARCH_BUILD_AS_LIBRARY=ON -DREDISEARCH_BUILD_SHARED=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON$EXTRA_CMAKE"

# REDISEARCH_SAN=address builds the `debug-asan` flavor (VecSim + RediSearch
# compiled with -fsanitize=address). Used by build/runtime/Dockerfile.asan so
# the embedded VectorSimilarity is instrumented; empty for normal builds.
SAN_ARG=""
if [ -n "${REDISEARCH_SAN:-}" ]; then
  SAN_ARG="SAN=${REDISEARCH_SAN}"
  # A sanitizer build compiles redisearch_rs's std with -Zbuild-std (build.sh uses
  # the nightly pinned in .rust-nightly), which needs the rust-src component.
  if [ -f .rust-nightly ]; then
    rustup toolchain install "$(cat .rust-nightly)" -c rust-src || true
  fi
fi

./build.sh BUILD_SEARCH_UNIT_TESTS=OFF $SAN_ARG
