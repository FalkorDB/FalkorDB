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
#   * JIT=1 (build) + GxB_JIT_PAUSE (runtime, set in matrix.rs init). The
#     build flag enables PreJIT compilation so the 188 baked-in kernels end
#     up in libgraphblas.a. The runtime flag restricts GraphBLAS to those
#     PreJIT kernels + the generic fallback path — no dlopen of cached or
#     freshly compiled .so files. This keeps us fork-safe (the original PR
#     #483 concern that motivated JIT_OFF) and avoids GxB_JIT_ERROR panics
#     in ops not covered by the C-derived PreJIT set (e.g. GrB_transpose
#     with GrB_DESC_RCT0). We diverge from FalkorDB C here, which uses
#     GxB_JIT_RUN; at RUN, GraphBLAS attempts cache loads for un-covered
#     ops and errors out when the cache is empty / no compiler is available.
#
# Two extras vs. plain upstream v10.3.1:
#
#   1. Apply build/graphblas/GB_control.patch — disables FP32/FP64/FC32/FC64
#      FactoryKernel families that FalkorDB query plans never hit. Matches the
#      tweak in FalkorDB C's vendored copy of GraphBLAS.
#
#   2. Vendored PreJIT/*.c kernels checked into build/graphblas/PreJIT/ —
#      these are harvested from running the rust port's full test suite
#      with the JIT engine on (see gen_prejit.sh). GraphBLAS's CMake globs
#      PreJIT/*.c and bakes them into libgraphblas.a so we get factory-
#      comparable speed for the operations our port actually executes
#      without runtime JIT compilation. Re-generate with gen_prejit.sh
#      when introducing new GraphBLAS op shapes or bumping GRAPHBLAS_VERSION.
#
# Harvest mode (FALKORDB_PREJIT_HARVEST=1, set by gen_prejit.sh):
#
#   * Skips copying the vendored PreJIT kernels into the GraphBLAS source
#     tree so the resulting libgraphblas.a has zero PreJIT — every op falls
#     through to the JIT engine, which compiles a fresh .c kernel into
#     ~/.SuiteSparse/GrBx.y.z/c/ for gen_prejit.sh to harvest.
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

# CLI: --skip-graphblas reuses the libgraphblas.a already installed under
# /usr/local (skips clone + 12-min compile + sudo install). Handy when
# iterating on the LAGraph cmake invocation alone.
SKIP_GRAPHBLAS=0
for arg in "$@"; do
    case "${arg}" in
        --skip-graphblas) SKIP_GRAPHBLAS=1 ;;
        *) echo "unknown arg: ${arg}" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPHBLAS_VERSION="${GRAPHBLAS_VERSION:-v10.3.1}"
LAGRAPH_VERSION="${LAGRAPH_VERSION:-v1.3.x}"

# Pinned FalkorDB C SHA — no longer used; kept here to mark the source of
# the original vendored PreJIT seed (commit captured 2026-05-27). We now
# vendor our own harvested kernels in build/graphblas/PreJIT/ via
# gen_prejit.sh, so changing this constant has no effect on the build.
FALKORDB_C_SHA="${FALKORDB_C_SHA:-7568688f358ef8227753dcdc37b30e7761ff07a6}"

# Harvest mode: when set, skip vendoring PreJIT kernels so libgraphblas.a
# falls through to the runtime JIT for every op. Used by gen_prejit.sh.
FALKORDB_PREJIT_HARVEST="${FALKORDB_PREJIT_HARVEST:-0}"

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

# Shared compiler flags applied to both GraphBLAS and LAGraph. All in a
# single COMMON_C_FLAGS bucket: we deliberately leave CMAKE_BUILD_TYPE
# unset so cmake never appends any CMAKE_C_FLAGS_<TYPE> variant on top
# of these — what you read here is exactly what hits the compile line.
#
#   -O3 -g -DNDEBUG                  : RelWithDebInfo-equivalent
#                                       optimization with debug symbols
#                                       (cmake's RelWithDebInfo defaults
#                                       to -O2 -g -DNDEBUG; we want -O3).
#   -fPIC                            : both libraries are statically linked
#                                       into libfalkordb.{so,dylib} (a
#                                       position-independent shared object),
#                                       so every translation unit must be
#                                       PIC.
#   -fno-stack-protector             : matches the FalkorDB C engine flags;
#                                       drops the stack-canary epilogue from
#                                       hot loops.
#   -Wno-incompatible-pointer-types  : clang-22 promoted this from warning to
#                                       error by default. v10.3.1's
#                                       GB_I_inverse.c trips it; LAGraph
#                                       stays quiet today but the flag is
#                                       harmless.
COMMON_C_FLAGS="-O3 -g -DNDEBUG -fPIC -fno-stack-protector -Wno-incompatible-pointer-types"
if [ -n "${INCLUDE_FLAG}" ]; then
    COMMON_C_FLAGS="${COMMON_C_FLAGS} ${INCLUDE_FLAG}"
fi

# Install prefix for libgraphblas.a + headers. Defaults to /usr/local
# (needs sudo on non-root hosts; install_cmd handles that). Set to a
# user-writable path (e.g. ~/.local) to skip sudo entirely — gen_prejit.sh
# uses this for its intermediate harvest build, paired with cargo
# build.rs's GRAPHBLAS_LIB_DIR env var.
GRAPHBLAS_INSTALL_PREFIX="${GRAPHBLAS_INSTALL_PREFIX:-/usr/local}"

install_cmd() {
    # Pre-create the install prefix so the -w check works correctly even
    # when the prefix doesn't exist yet (e.g. fresh harvest run).
    mkdir -p "${GRAPHBLAS_INSTALL_PREFIX}" 2>/dev/null || true
    if [ -w "${GRAPHBLAS_INSTALL_PREFIX}" ] || [ "$(id -u)" -eq 0 ]; then
        cmake --install .
    else
        sudo cmake --install .
    fi
}

if [ "${SKIP_GRAPHBLAS}" -eq 0 ]; then
    # --- clone GraphBLAS upstream ---------------------------------------------
    rm -rf GraphBLAS
    git clone --branch "${GRAPHBLAS_VERSION}" --single-branch --depth 1 \
        https://github.com/DrTimothyAldenDavis/GraphBLAS.git

    # --- apply GB_control.h customization -------------------------------------
    git -C GraphBLAS apply "${SCRIPT_DIR}/build/graphblas/GB_control.patch"

    # --- vendor PreJIT kernels from build/graphblas/PreJIT/ -------------------
    # In normal mode, copy our harvested .c kernels into the GraphBLAS
    # source tree so CMake bakes them statically into libgraphblas.a.
    # In harvest mode (FALKORDB_PREJIT_HARVEST=1), skip this so the JIT
    # engine has to compile every kernel into ~/.SuiteSparse/.../c/ where
    # gen_prejit.sh can collect them.
    if [ "${FALKORDB_PREJIT_HARVEST}" -eq 1 ]; then
        echo "FALKORDB_PREJIT_HARVEST=1: skipping PreJIT vendoring (harvest mode)"
    else
        PREJIT_VENDOR_DIR="${SCRIPT_DIR}/build/graphblas/PreJIT"
        shopt -s nullglob
        prejit_files=( "${PREJIT_VENDOR_DIR}"/GB_jit_*.c )
        shopt -u nullglob
        if [ "${#prejit_files[@]}" -gt 0 ]; then
            cp "${prejit_files[@]}" GraphBLAS/PreJIT/
            echo "vendored ${#prejit_files[@]} PreJIT kernels from ${PREJIT_VENDOR_DIR}"
        else
            echo "no PreJIT kernels in ${PREJIT_VENDOR_DIR} – run gen_prejit.sh to populate"
        fi
    fi

    # --- build GraphBLAS ------------------------------------------------------
    mkdir -p GraphBLAS/build
    (
        cd GraphBLAS/build
        cmake \
            -DCMAKE_INSTALL_PREFIX="${GRAPHBLAS_INSTALL_PREFIX}" \
            -DSUITESPARSE_USE_FORTRAN=OFF \
            -DBUILD_STATIC_LIBS=ON \
            -DBUILD_SHARED_LIBS=OFF \
            -DGRAPHBLAS_COMPACT=OFF \
            -DGRAPHBLAS_BUILD_STATIC_LIBS=ON \
            -DBUILD_TESTING=OFF \
            -DGRAPHBLAS_USE_JIT=1 \
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
            -DCMAKE_C_FLAGS="${COMMON_C_FLAGS}" \
            -DCMAKE_CXX_FLAGS="${COMMON_C_FLAGS}" \
            ${CMAKE_COMPILER_ARGS[@]+"${CMAKE_COMPILER_ARGS[@]}"} \
            ${OPENMP_CMAKE_ARGS[@]+"${OPENMP_CMAKE_ARGS[@]}"} \
            ..
        cmake --build . -j"${JOBS}"
        install_cmd
    )
    rm -rf GraphBLAS
else
    echo "--skip-graphblas: reusing ${GRAPHBLAS_INSTALL_PREFIX}/lib/libgraphblas.a"
    test -f "${GRAPHBLAS_INSTALL_PREFIX}/lib/libgraphblas.a" || {
        echo "ERROR: ${GRAPHBLAS_INSTALL_PREFIX}/lib/libgraphblas.a not found; run without --skip-graphblas first" >&2
        exit 1
    }
fi

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
        -DGRAPHBLAS_INCLUDE_DIR="${GRAPHBLAS_INSTALL_PREFIX}/include/suitesparse" \
        -DGRAPHBLAS_LIBRARY="${GRAPHBLAS_INSTALL_PREFIX}/lib/libgraphblas.a" \
        -DGraphBLAS_DIR="${GRAPHBLAS_INSTALL_PREFIX}/lib/cmake/GraphBLAS" \
        -DCMAKE_PREFIX_PATH="${GRAPHBLAS_INSTALL_PREFIX}" \
        -DSUITESPARSE_USE_FORTRAN=OFF \
        -DCMAKE_C_FLAGS="${COMMON_C_FLAGS}" \
        -DCMAKE_CXX_FLAGS="${COMMON_C_FLAGS}" \
        ${CMAKE_COMPILER_ARGS[@]+"${CMAKE_COMPILER_ARGS[@]}"} \
        ${OPENMP_CMAKE_ARGS[@]+"${OPENMP_CMAKE_ARGS[@]}"} \
        ..
    cmake --build . -j"${JOBS}"
)

mkdir -p "${LAGRAPH_INSTALL_DIR}"
cp LAGraph/build/src/liblagraph.a "${LAGRAPH_INSTALL_DIR}/"
cp LAGraph/build/experimental/liblagraphx.a "${LAGRAPH_INSTALL_DIR}/"
cp LAGraph/include/LAGraph.h "${LAGRAPH_INSTALL_DIR}/"
cp LAGraph/include/LAGraphX.h "${LAGRAPH_INSTALL_DIR}/"
rm -rf LAGraph
