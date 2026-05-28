#!/bin/bash
# Build SuiteSparse:GraphBLAS v10.3.1 + LAGraph v1.2.1 for FalkorDB.
#
# Single script for both deps because LAGraph statically links against the
# graphblas archive we just installed; keeping them together avoids drift in
# the OpenMP / compiler / install-prefix flags they must share.
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
# Platform handling (so a Mac dev `./graphblas.sh` just works):
#
#   * Linux + /opt/libomp present (Docker toolchain image): force GraphBLAS to
#     resolve OpenMP against our static libomp.a so libfalkordb.so embeds the
#     libomp ABI rather than dynamically depending on libgomp.so.1.
#   * Everywhere else (macOS, local Linux dev): rely on whatever compiler is
#     on PATH (or CC/CXX env) and let cmake's find_package(OpenMP) auto-detect.
#     On macOS with homebrew LLVM, set `export CC=$(brew --prefix llvm)/bin/clang`
#     before invoking. CI/Docker is the source of truth for the self-contained .so.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPHBLAS_VERSION="${GRAPHBLAS_VERSION:-v10.3.1}"
LAGRAPH_VERSION="${LAGRAPH_VERSION:-v1.2.1}"

# Pinned FalkorDB C SHA whose deps/GraphBLAS/PreJIT/ kernel set we vendor.
# Bump in lock-step with PreJIT regenerations on the C side. Capturing the
# current main HEAD as of 2026-05-27.
FALKORDB_C_SHA="${FALKORDB_C_SHA:-7568688f358ef8227753dcdc37b30e7761ff07a6}"

JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)}"

# Compiler: honour CC/CXX from the environment, else let cmake pick. The
# Docker toolchain image exports CC=clang-22 / CXX=clang++-22.
CMAKE_COMPILER_ARGS=()
if [ -n "${CC:-}" ]; then CMAKE_COMPILER_ARGS+=(-DCMAKE_C_COMPILER="${CC}"); fi
if [ -n "${CXX:-}" ]; then CMAKE_COMPILER_ARGS+=(-DCMAKE_CXX_COMPILER="${CXX}"); fi

# OpenMP: only force the static-libomp wiring inside the Docker toolchain
# image (where build/libomp.sh has produced /opt/libomp/lib/libomp.a). On
# every other host let cmake's find_package(OpenMP) auto-detect.
LIBOMP_PREFIX="${LIBOMP_PREFIX:-/opt/libomp}"
OPENMP_CMAKE_ARGS=()
INCLUDE_FLAG=""
if [ -f "${LIBOMP_PREFIX}/lib/libomp.a" ]; then
    INCLUDE_FLAG="-I${LIBOMP_PREFIX}/include"
    OPENMP_CMAKE_ARGS=(
        -DOpenMP_C_FLAGS=-fopenmp=libomp
        -DOpenMP_CXX_FLAGS=-fopenmp=libomp
        -DOpenMP_C_LIB_NAMES=omp
        -DOpenMP_CXX_LIB_NAMES=omp
        -DOpenMP_omp_LIBRARY="${LIBOMP_PREFIX}/lib/libomp.a"
    )
fi

install_cmd() {
    if [ "$(id -u)" -eq 0 ]; then
        cmake --install .
    else
        sudo cmake --install .
    fi
}

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
    git checkout --detach "${FALKORDB_C_SHA}"
)
cp FalkorDB-prejit/deps/GraphBLAS/PreJIT/GB_jit_*.c GraphBLAS/PreJIT/
rm -rf FalkorDB-prejit
echo "vendored $(ls GraphBLAS/PreJIT/GB_jit_*.c | wc -l) PreJIT kernels from FalkorDB@${FALKORDB_C_SHA}"

# --- build GraphBLAS ----------------------------------------------------------
mkdir -p GraphBLAS/build
(
    cd GraphBLAS/build
    cmake \
        -DSUITESPARSE_USE_FORTRAN=OFF \
        -DBUILD_STATIC_LIBS=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DGRAPHBLAS_COMPACT=OFF \
        -DGRAPHBLAS_BUILD_STATIC_LIBS=ON \
        -DBUILD_TESTING=OFF \
        -DGRAPHBLAS_USE_JIT=0 \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_C_FLAGS="-fPIC -fno-stack-protector -Wno-incompatible-pointer-types ${INCLUDE_FLAG}" \
        -DCMAKE_CXX_FLAGS="-fPIC -fno-stack-protector -Wno-incompatible-pointer-types ${INCLUDE_FLAG}" \
        -DCMAKE_C_FLAGS_RELWITHDEBINFO="-O3 -g -DNDEBUG -fPIC -fno-stack-protector -Wno-incompatible-pointer-types" \
        -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -g -DNDEBUG -fPIC -fno-stack-protector -Wno-incompatible-pointer-types" \
        "${CMAKE_COMPILER_ARGS[@]}" \
        "${OPENMP_CMAKE_ARGS[@]}" \
        ..
    cmake --build . --config RelWithDebInfo -j"${JOBS}"
    install_cmd
)
rm -rf GraphBLAS

# --- build LAGraph ------------------------------------------------------------
# LAGraph links against the just-installed libgraphblas.a. Output goes to
# ./lagraph_lib (read by graph/build.rs via ../lagraph_lib relative to the
# graph crate manifest).
LAGRAPH_INSTALL_DIR="${LAGRAPH_INSTALL_DIR:-${SCRIPT_DIR}/lagraph_lib}"

rm -rf LAGraph
git clone --branch "${LAGRAPH_VERSION}" --single-branch --depth 1 \
    https://github.com/GraphBLAS/LAGraph.git
mkdir -p LAGraph/build
(
    cd LAGraph/build
    cmake \
        -DBUILD_STATIC_LIBS=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DLIBRARY_ONLY=ON \
        -DBUILD_TESTING=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DGRAPHBLAS_INCLUDE_DIR=/usr/local/include/suitesparse \
        -DGRAPHBLAS_LIBRARY=/usr/local/lib/libgraphblas.a \
        -DSUITESPARSE_USE_FORTRAN=OFF \
        -DCMAKE_C_FLAGS="${INCLUDE_FLAG}" \
        -DCMAKE_CXX_FLAGS="${INCLUDE_FLAG}" \
        "${CMAKE_COMPILER_ARGS[@]}" \
        "${OPENMP_CMAKE_ARGS[@]}" \
        ..
    cmake --build . --config Release -j"${JOBS}"
)

mkdir -p "${LAGRAPH_INSTALL_DIR}"
cp LAGraph/build/src/liblagraph.a "${LAGRAPH_INSTALL_DIR}/"
cp LAGraph/build/experimental/liblagraphx.a "${LAGRAPH_INSTALL_DIR}/"
cp LAGraph/include/LAGraph.h "${LAGRAPH_INSTALL_DIR}/"
cp LAGraph/include/LAGraphX.h "${LAGRAPH_INSTALL_DIR}/"
rm -rf LAGraph
