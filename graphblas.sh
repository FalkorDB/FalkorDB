#!/bin/bash
# Build SuiteSparse:GraphBLAS v10.3.1 for FalkorDB.
#
# Mirrors the build the FalkorDB C project uses (see
# https://github.com/FalkorDB/FalkorDB/blob/master/build.sh `build_graphblas`):
#
#   * GRAPHBLAS_COMPACT=OFF — keep FactoryKernels enabled. Previous version
#     of this script set =1 which disables FactoryKernels entirely; that was
#     incompatible with the PreJIT-baking strategy below.
#   * RelWithDebInfo + -O3 -fPIC -fno-stack-protector — matches C engine.
#   * JIT=0 — module disables JIT at runtime (PR #483) to avoid dlopen-vs-fork
#     deadlocks. Performance is recovered by baking PreJIT kernels in at build
#     time instead.
#
# Two extras vs. plain upstream v10.3.1:
#
#   1. Apply build/graphblas/GB_control.patch — disables FP32/FP64/FC32/FC64
#      FactoryKernel families that FalkorDB query plans never hit. Matches the
#      tweak in FalkorDB C's vendored copy of GraphBLAS.
#
#   2. Sparse-checkout 188 vendored PreJIT/*.c kernels from the FalkorDB C
#      repo at a pinned SHA. Upstream's PreJIT directory contains only a
#      README; these kernels are a workload-specific cache captured by
#      running the C engine's test suite with JIT on and committing the
#      resulting .c files. GraphBLAS's CMake globs PreJIT/*.c and bakes
#      them into libgraphblas.a, so we get factory-comparable speed for the
#      operations FalkorDB actually executes without runtime JIT.
#
# OpenMP: clang-22's -fopenmp default looks for libomp shared object. We
# don't ship one in the toolchain image anymore (libomp-22-dev removed) —
# instead build/libomp.sh produced /opt/libomp/lib/libomp.a. Point cmake at
# our static archive + headers so GraphBLAS picks up the same libomp ABI
# that libfalkordb.so ultimately statically links against. Without this,
# GraphBLAS's find_package(OpenMP) would silently fall back to GCC libgomp
# (system /lib/.../libgomp.so.1), which is a different ABI and would
# re-introduce a libgomp.so.1 runtime dependency on libfalkordb.so.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPHBLAS_VERSION="${GRAPHBLAS_VERSION:-v10.3.1}"

# Pinned FalkorDB C SHA whose deps/GraphBLAS/PreJIT/ kernel set we vendor.
# Bump in lock-step with PreJIT regenerations on the C side. Capturing the
# current main HEAD as of 2026-05-27.
FALKORDB_C_SHA="${FALKORDB_C_SHA:-7568688f358ef8227753dcdc37b30e7761ff07a6}"

LIBOMP_PREFIX="${LIBOMP_PREFIX:-/opt/libomp}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 2)}"

# --- clone GraphBLAS upstream -------------------------------------------------
rm -rf GraphBLAS
git clone --branch "${GRAPHBLAS_VERSION}" --single-branch --depth 1 \
    https://github.com/DrTimothyAldenDavis/GraphBLAS.git

# --- apply GB_control.h customization -----------------------------------------
git -C GraphBLAS apply "${SCRIPT_DIR}/build/graphblas/GB_control.patch"

# --- pull vendored PreJIT kernels from FalkorDB C -----------------------------
rm -rf FalkorDB-prejit
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/FalkorDB/FalkorDB.git FalkorDB-prejit
(
    cd FalkorDB-prejit
    git sparse-checkout set --no-cone deps/GraphBLAS/PreJIT
    git fetch --depth 1 origin "${FALKORDB_C_SHA}"
    git checkout "${FALKORDB_C_SHA}" --
)
cp FalkorDB-prejit/deps/GraphBLAS/PreJIT/GB_jit_*.c GraphBLAS/PreJIT/
rm -rf FalkorDB-prejit
echo "vendored $(ls GraphBLAS/PreJIT/GB_jit_*.c | wc -l) PreJIT kernels from FalkorDB@${FALKORDB_C_SHA}"

# --- build --------------------------------------------------------------------
mkdir -p GraphBLAS/build
cd GraphBLAS/build

CMAKE_ARGS=(
    -DSUITESPARSE_USE_FORTRAN=OFF
    -DBUILD_STATIC_LIBS=ON
    -DBUILD_SHARED_LIBS=OFF
    -DGRAPHBLAS_COMPACT=OFF
    -DGRAPHBLAS_BUILD_STATIC_LIBS=ON
    -DBUILD_TESTING=OFF
    -DGRAPHBLAS_USE_JIT=0
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DCMAKE_C_COMPILER=clang-22
    -DCMAKE_CXX_COMPILER=clang++-22
    -DCMAKE_C_FLAGS="-fPIC -fno-stack-protector -Wno-incompatible-pointer-types -I${LIBOMP_PREFIX}/include"
    -DCMAKE_CXX_FLAGS="-fPIC -fno-stack-protector -Wno-incompatible-pointer-types -I${LIBOMP_PREFIX}/include"
    "-DCMAKE_C_FLAGS_RELWITHDEBINFO=-O3 -g -DNDEBUG -fPIC -fno-stack-protector -Wno-incompatible-pointer-types"
    "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O3 -g -DNDEBUG -fPIC -fno-stack-protector -Wno-incompatible-pointer-types"
    # Force find_package(OpenMP) onto our libomp.a so emitted references are
    # libomp-ABI, not libgomp-ABI. The final libfalkordb.so link resolves these
    # against /opt/libomp/lib/libomp.a (see graph/build.rs).
    -DOpenMP_C_FLAGS=-fopenmp=libomp
    -DOpenMP_CXX_FLAGS=-fopenmp=libomp
    -DOpenMP_C_LIB_NAMES=omp
    -DOpenMP_CXX_LIB_NAMES=omp
    -DOpenMP_omp_LIBRARY="${LIBOMP_PREFIX}/lib/libomp.a"
)

cmake "${CMAKE_ARGS[@]}" ..
cmake --build . --config RelWithDebInfo -j"${JOBS}"
if [ "$(id -u)" -eq 0 ]; then
    cmake --install .
else
    sudo cmake --install .
fi

cd ../..
rm -rf GraphBLAS
