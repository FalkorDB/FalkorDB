#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
REDISEARCH_BRANCH="falkordb/llapi-extensions-8.6"
REDISEARCH_DIR="$ROOT/redisearch/RediSearch"

mkdir -p "$ROOT/redisearch"
if [ ! -d "$REDISEARCH_DIR/.git" ]; then
  git clone --recurse-submodules --branch "$REDISEARCH_BRANCH" --single-branch --depth 1 \
    https://github.com/FalkorDB/RediSearch.git "$REDISEARCH_DIR"
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
