#!/bin/bash
# Build LLVM libomp as a static archive for the toolchain image.
#
# apt.llvm.org's libomp-23-dev ships only libomp.so / libomp.so.5 (no .a),
# so we compile the runtime ourselves from llvm-project. Output:
#   /opt/libomp/lib/libomp.a
#   /opt/libomp/include/omp.h
#
# This lets graph/build.rs link omp statically and removes libomp.so.5 as a
# runtime dependency of libfalkordb.so. The toolchain image (and any downstream
# stage that consumes /opt/libomp) gets a single self-contained tree to point
# cmake at via LIBOMP_ROOT.
#
# Network footprint: --filter=blob:none + sparse-checkout of openmp/cmake/
# runtimes only — ~30 MB instead of ~1.5 GB for a full llvm-project clone.
#
# Compiler: honours CC/CXX from the environment; otherwise the Docker image
# exports clang-${CLANG_MAJOR}, and a local mac dev would set
# CC=$(brew --prefix llvm)/bin/clang.

set -euo pipefail

# Resolve the libomp source release. Priority:
#   1. explicit positional arg ($1)
#   2. auto-detect from ${CC:-clang} --version
# Auto-detection makes the libomp ABI track whatever clang is on PATH,
# avoiding silent drift between the clang we build with and the libomp
# we statically link in.
detect_llvm_version() {
    local cc_bin="${CC:-clang}"
    "${cc_bin}" --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1
}
LLVMORG_VERSION="${1:-$(detect_llvm_version)}"
if [ -z "${LLVMORG_VERSION}" ]; then
    echo "ERROR: could not detect LLVM version; pass it explicitly: ./build/libomp.sh 23.1.2" >&2
    exit 1
fi
echo "libomp.sh: requested llvmorg-${LLVMORG_VERSION}"

# apt.llvm.org sometimes ships a clang point release (e.g. 23.1.3) BEFORE the
# llvm-project git tag of the same name has been cut upstream. In that
# window a literal `git clone --branch llvmorg-23.1.3` aborts with "Remote
# branch not found" and the toolchain image build fails.
#
# Resolve the requested X.Y.Z to the highest existing patch ≤ Z for the
# same X.Y. This keeps us ABI-compatible with the installed clang (libomp
# only needs to match the major/minor — the patch release affects bug
# fixes, not the OpenMP runtime ABI) while surviving the upstream lag.
LLVM_MAJOR_MINOR="${LLVMORG_VERSION%.*}"
LLVM_PATCH="${LLVMORG_VERSION##*.}"
RESOLVED_TAG=$(
    git ls-remote --tags --refs https://github.com/llvm/llvm-project.git \
        "refs/tags/llvmorg-${LLVM_MAJOR_MINOR}.*" 2>/dev/null \
    | awk -F'refs/tags/llvmorg-' '{print $2}' \
    | awk -F. -v want="${LLVM_PATCH}" '$3 ~ /^[0-9]+$/ && $3+0 <= want+0 {print}' \
    | sort -t. -k3,3n \
    | tail -1
)
if [ -z "${RESOLVED_TAG}" ]; then
    # No patch ≤ requested exists — fall back to the highest patch for the
    # major.minor channel (covers the case where apt.llvm.org regresses).
    RESOLVED_TAG=$(
        git ls-remote --tags --refs https://github.com/llvm/llvm-project.git \
            "refs/tags/llvmorg-${LLVM_MAJOR_MINOR}.*" 2>/dev/null \
        | awk -F'refs/tags/llvmorg-' '{print $2}' \
        | awk -F. '$3 ~ /^[0-9]+$/' \
        | sort -t. -k3,3n \
        | tail -1
    )
fi
if [ -z "${RESOLVED_TAG}" ]; then
    echo "ERROR: no llvmorg-${LLVM_MAJOR_MINOR}.* tag found upstream" >&2
    exit 1
fi
if [ "${RESOLVED_TAG}" != "${LLVMORG_VERSION}" ]; then
    echo "libomp.sh: llvmorg-${LLVMORG_VERSION} not yet tagged upstream; using llvmorg-${RESOLVED_TAG} instead"
fi
LLVMORG_VERSION="${RESOLVED_TAG}"
echo "libomp.sh: building llvmorg-${LLVMORG_VERSION}"

PREFIX="${PREFIX:-/opt/libomp}"
SRC_DIR="${SRC_DIR:-/tmp/llvm-project}"
BUILD_DIR="${SRC_DIR}/build-openmp"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)}"

CMAKE_COMPILER_ARGS=()
if [ -n "${CC:-}" ]; then CMAKE_COMPILER_ARGS+=(-DCMAKE_C_COMPILER="${CC}"); fi
if [ -n "${CXX:-}" ]; then CMAKE_COMPILER_ARGS+=(-DCMAKE_CXX_COMPILER="${CXX}"); fi

rm -rf "${SRC_DIR}"
git clone \
    --depth 1 \
    --branch "llvmorg-${LLVMORG_VERSION}" \
    --filter=blob:none \
    --sparse \
    https://github.com/llvm/llvm-project.git \
    "${SRC_DIR}"

cd "${SRC_DIR}"
# The runtimes build pulls CMake modules and lit config from llvm/ as well.
git sparse-checkout set --no-cone openmp cmake runtimes llvm/cmake llvm/utils third-party

# Build through runtimes/: LLVM 23 removed openmp/'s standalone mode
# ("The legacy standalone build mode has been removed"), and runtimes/ works
# for earlier majors too. Per-target dirs off keeps libomp.a in ${PREFIX}/lib.
cmake \
    -S runtimes \
    -B "${BUILD_DIR}" \
    -DLLVM_ENABLE_RUNTIMES=openmp \
    -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBOMP_ENABLE_SHARED=OFF \
    -DLIBOMP_OMPD_SUPPORT=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DOPENMP_ENABLE_OMPT_TOOLS=OFF \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    ${CMAKE_COMPILER_ARGS[@]+"${CMAKE_COMPILER_ARGS[@]}"}

cmake --build "${BUILD_DIR}" -j"${JOBS}" --target install

# Sanity: archive present, no .so produced.
test -f "${PREFIX}/lib/libomp.a"
test -f "${PREFIX}/include/omp.h"
# Forbid only libomp.so* / .dylib (we only ship libomp.a).
if find "${PREFIX}/lib" -maxdepth 1 \( -name 'libomp.so*' -o -name 'libomp.*dylib' \) | grep -q .; then
    echo "unexpected shared libomp produced under ${PREFIX}" >&2
    find "${PREFIX}/lib" -maxdepth 1 \( -name 'libomp.so*' -o -name 'libomp.*dylib' \) >&2
    exit 1
fi

rm -rf "${SRC_DIR}"
