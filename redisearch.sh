#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
# Pin RediSearch to a specific commit on falkordb/llapi-extensions-8.6 rather
# than tracking the branch tip. The commit lives in this file, which is COPY'd
# into build/Dockerfile, so bumping REDISEARCH_REF busts the toolchain image's
# cached `RUN redisearch.sh` layer and keeps the build reproducible. To pick up
# new RediSearch work, update REDISEARCH_REF to the new commit (check-files in
# .github/workflows/rust-pr.yml rebuilds the toolchain when this file changes).
REDISEARCH_BRANCH="falkordb/llapi-extensions-8.6"
REDISEARCH_REF="ac45247834fec495f4d3ec76f337e3709260fdd5"
REDISEARCH_DIR="$ROOT/redisearch/RediSearch"

mkdir -p "$ROOT/redisearch"
if [ ! -d "$REDISEARCH_DIR/.git" ]; then
  # Shallow-fetch just the pinned commit (GitHub allows fetching a reachable SHA).
  git init -q "$REDISEARCH_DIR"
  git -C "$REDISEARCH_DIR" remote add origin https://github.com/FalkorDB/RediSearch.git
  git -C "$REDISEARCH_DIR" fetch -q --depth 1 origin "$REDISEARCH_REF"
  git -C "$REDISEARCH_DIR" checkout -q FETCH_HEAD
  git -C "$REDISEARCH_DIR" submodule update --init --recursive --depth 1
else
  # Reusing an existing checkout (fast local iteration; CI always clones fresh
  # because redisearch/ is .dockerignore'd). Warn loudly if it isn't on the
  # pinned ref so a stale or branch-switched clone doesn't silently build
  # against the wrong RediSearch ABI. We deliberately do NOT auto-reset — that
  # would clobber local RediSearch work; remove the directory to force a clean
  # re-fetch.
  current_ref="$(git -C "$REDISEARCH_DIR" rev-parse HEAD 2>/dev/null || echo '?')"
  if [ "$current_ref" != "$REDISEARCH_REF" ]; then
    echo "WARNING: $REDISEARCH_DIR is at '$current_ref', expected '$REDISEARCH_REF'." >&2
    echo "         Remove that directory to re-fetch, or check out the expected ref." >&2
  fi
fi

cd "$REDISEARCH_DIR"

# VecSim promotes some warnings to errors; relax for our toolchain.
if [[ "$(uname -s)" == "Darwin" ]]; then
  sed -i '' 's/-Werror//g' deps/VectorSimilarity/src/VecSim/CMakeLists.txt
else
  sed -i 's/-Werror//g' deps/VectorSimilarity/src/VecSim/CMakeLists.txt
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
