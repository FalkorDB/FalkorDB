#!/bin/bash
# Build the standalone Tensor micro-benchmark.
#
# Sources come from the read-only master worktree ($SRC); everything else is
# linked out of the pre-existing C build tree ($BIN) - nothing there is written.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# Both inputs are overridable, because neither lives in this repo's tree:
#
#   SRC  a checkout of `master` (the C engine's sources). A git worktree is the
#        cheapest way to get one:  git worktree add /tmp/cmaster master
#   BIN  a C build tree, for `libfalkordb_static.a` and its GraphBLAS. The
#        harness links the *shipped* archive rather than recompiling tensor.c,
#        so what it measures is what ships.
#
# Defaults suit a macOS arm64 checkout with `make` already run.
SRC="${SRC:-$(cd "$HERE/../../.." && pwd)/../cmaster}"
BIN="${BIN:-$(cd "$HERE/../../.." && pwd)/bin/macos-arm64v8-release}"
CC="${CC:-/opt/homebrew/opt/llvm/bin/clang}"

if [ ! -d "$SRC/src/graph/tensor" ]; then
	echo "SRC=$SRC is not a master checkout (no src/graph/tensor)." >&2
	echo "Point SRC at one:  git worktree add /tmp/cmaster master" >&2
	exit 1
fi
if [ ! -f "$BIN/libfalkordb_static.a" ]; then
	echo "BIN=$BIN has no libfalkordb_static.a — build the C engine first." >&2
	exit 1
fi
OBJ="$HERE/obj"
mkdir -p "$OBJ"

# NOTE: deps/xxHash and deps/RediSearch are git submodules and are NOT populated
# in this worktree. xxhash.h is taken from the copy vendored inside
# deps/GraphBLAS (only needed to compile value.h's hashing decls; the harness
# never calls XXH*).
INC="-I$SRC -I$SRC/src -I$SRC/deps -I$SRC/deps/rax -I$SRC/deps/GraphBLAS/xxHash \
-I$SRC/deps/quickjs -I$SRC/deps/utf8proc -I$SRC/deps/oniguruma \
-I$SRC/deps/RediSearch/src -I$SRC/deps/LAGraph/include \
-I$SRC/deps/GraphBLAS/Include -I$SRC/deps/libcurl/include/curl \
-I$SRC/deps/libcypher-parser/lib/src -I$BIN/libcypher-parser/lib/src \
-isystem /opt/homebrew/opt/openssl@3/include"

DEFS="-DREDISMODULE_EXPERIMENTAL_API -DREDIS_MODULE_TARGET -DXXH_STATIC_LINKING_ONLY -D_GNU_SOURCE"
CFLAGS="-Wno-gnu-zero-variadic-macro-arguments -Wno-error=incompatible-pointer-types -fopenmp=libomp -O3 -g -DNDEBUG \
-std=gnu11 -arch arm64 -mmacosx-version-min=12.0 -fno-strict-aliasing -UNDEBUG"

# Only the harness is compiled from source. tensor.c / tensor_iterator.c /
# delta_matrix/*.c come out of $BIN/libfalkordb_static.a, which was built from
# the same commit-era sources against the same GraphBLAS 10.3.1 that
# $BIN/GraphBLAS/libgraphblas.a provides -- so struct layouts agree.
# (Compiling delta_matrix/*.c here is not possible in this worktree: several of
# them include globals.h -> graphcontext.h -> index/index_field.h ->
# redisearch_api.h, and deps/RediSearch is an unpopulated submodule.)
SOURCES=("$HERE/tensor_bench.c")

OBJS=()
for f in "${SOURCES[@]}"; do
  o="$OBJ/$(basename "$f" .c).o"
  echo "CC $(basename "$f")"
  $CC $DEFS $INC $CFLAGS -c "$f" -o "$o"
  OBJS+=("$o")
done

echo "LD tensor_bench"
$CC -arch arm64 -mmacosx-version-min=12.0 -fopenmp=libomp \
  "${OBJS[@]}" \
  "$@" \
  "$BIN/libfalkordb_static.a" \
  "$BIN/GraphBLAS/libgraphblas.a" \
  "$BIN/LAGraph/src/liblagraph.a" \
  "$BIN/LAGraph/experimental/liblagraphx.a" \
  "$BIN/xxHash/libxxhash.a" \
  "$BIN/rax/librax.a" \
  "$BIN/utf8proc/libutf8proc.a" \
  "$BIN/oniguruma/libonig.a" \
  "$BIN/quickjs/libquickjs.a" \
  "$BIN/libcsv/.libs/libcsv.a" \
  "$BIN/libcurl/lib/.libs/libcurl.a" \
  "$BIN/libcypher-parser/lib/src/.libs/libcypher-parser.a" \
  "$BIN/search-static/libredisearch-static.a" \
  "$BIN/search-static/deps/VectorSimilarity/src/VecSim/libVectorSimilarity.a" \
  "$BIN/search-static/deps/VectorSimilarity/src/VecSim/spaces/libVectorSimilaritySpaces.a" \
  "$BIN/search-static/deps/VectorSimilarity/src/VecSim/spaces/libVectorSimilaritySpaces_no_optimization.a" \
  -L/opt/homebrew/opt/openssl@3/lib -lssl -lcrypto \
  -framework CoreFoundation -framework SystemConfiguration \
  -lc++ -lomp -L/opt/homebrew/opt/llvm/lib \
  -o "$HERE/tensor_bench"
echo "built $HERE/tensor_bench"
