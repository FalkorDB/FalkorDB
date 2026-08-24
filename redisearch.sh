#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

# RediSearch lives at deps/RediSearch as a git submodule, the same layout the C
# engine uses on `master`. The committed gitlink is the source of truth for a
# git checkout; REDISEARCH_REF below is the same commit spelled out for builds
# that only have this script and no repository (the Docker toolchain images COPY
# redisearch.sh alone, so bumping REDISEARCH_REF busts their cached
# `RUN redisearch.sh` layer and keeps those builds reproducible).
#
# To move to a new RediSearch commit, do BOTH:
#   git -C deps/RediSearch fetch origin && git -C deps/RediSearch checkout <sha>
#   git add deps/RediSearch          # bump the gitlink
#   ...and set REDISEARCH_REF=<sha> here
# The two are checked against each other below, so a half-done bump fails loudly
# instead of building a different RediSearch locally than in Docker.
REDISEARCH_BRANCH="falkordb/llapi-extensions-8.6"
REDISEARCH_REF="ac45247834fec495f4d3ec76f337e3709260fdd5"
REDISEARCH_DIR="$ROOT/deps/RediSearch"

# The committed gitlink, when we are in a repository at all (empty otherwise).
gitlink_ref=""
if git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  # From the index, not HEAD, so a staged submodule bump is what gets checked.
  gitlink_ref="$(git -C "$ROOT" ls-files -s -- deps/RediSearch 2>/dev/null | awk '{print $2}')"
fi

if [ -n "$gitlink_ref" ] && [ "$gitlink_ref" != "$REDISEARCH_REF" ]; then
  echo "ERROR: deps/RediSearch gitlink ($gitlink_ref) != REDISEARCH_REF ($REDISEARCH_REF)." >&2
  echo "       These must agree: the gitlink drives submodule builds, REDISEARCH_REF" >&2
  echo "       drives the Docker toolchain images. Update REDISEARCH_REF in $0." >&2
  exit 1
fi

if [ ! -d "$REDISEARCH_DIR/.git" ]; then
  # Bootstrap the submodule path ourselves rather than via `git submodule
  # update`, and identically whether or not there is a repository here (the
  # Docker toolchain images COPY only this script). Two reasons:
  #   * `git submodule update --depth 1` shallow-clones the *default branch* and
  #     then cannot check out the pinned commit, which is usually not its tip;
  #     without --depth it clones RediSearch's full history, which is large and
  #     needless in CI.
  #   * a shallow fetch of the exact SHA lands the tree at the gitlink, so
  #     `git submodule status` is clean afterwards.
  # A developer who wants full RediSearch history can instead run
  # `git submodule update --init --recursive -- deps/RediSearch` before this
  # script; it sees the populated checkout below and leaves it alone.
  mkdir -p "$REDISEARCH_DIR"
  git init -q "$REDISEARCH_DIR"
  git -C "$REDISEARCH_DIR" remote add origin https://github.com/FalkorDB/RediSearch.git
  # GitHub allows fetching a reachable SHA, so no branch fetch is needed. The
  # branch name is recorded in .gitmodules (and REDISEARCH_BRANCH) for humans.
  git -C "$REDISEARCH_DIR" fetch -q --depth 1 origin "$REDISEARCH_REF"
  git -C "$REDISEARCH_DIR" checkout -q FETCH_HEAD
  git -C "$REDISEARCH_DIR" submodule update --init --recursive --depth 1
else
  # Reusing an existing checkout (fast local iteration, and the case that makes
  # a submodule worth having: edit deps/RediSearch in place, build, test, then
  # commit the gitlink bump). Warn rather than reset if it isn't on the pinned
  # ref -- a reset would clobber local RediSearch work -- so a stale or
  # branch-switched clone doesn't silently build against the wrong ABI.
  current_ref="$(git -C "$REDISEARCH_DIR" rev-parse HEAD 2>/dev/null || echo '?')"
  if [ "$current_ref" != "$REDISEARCH_REF" ]; then
    echo "WARNING: $REDISEARCH_DIR is at '$current_ref', expected '$REDISEARCH_REF'." >&2
    echo "         Building it anyway. To return to the pinned ref:" >&2
    echo "         git -C $ROOT submodule update --init --recursive -- deps/RediSearch" >&2
  fi
  # A checkout that predates a nested-submodule addition (or a `git submodule
  # update` without --recursive) leaves VectorSimilarity et al. empty.
  git -C "$REDISEARCH_DIR" submodule update --init --recursive --depth 1
fi

cd "$REDISEARCH_DIR"

# VecSim promotes some warnings to errors; relax for our toolchain. Restore the
# file afterwards so the submodule does not show up as dirty in the parent repo.
VECSIM_CMAKE="deps/VectorSimilarity/src/VecSim/CMakeLists.txt"
restore_vecsim() {
  git -C deps/VectorSimilarity checkout -- src/VecSim/CMakeLists.txt 2>/dev/null || true
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
