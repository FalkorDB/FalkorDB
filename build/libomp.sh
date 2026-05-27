#!/bin/bash
# Build LLVM libomp as a static archive for the toolchain image.
#
# apt.llvm.org's libomp-22-dev ships only libomp.so / libomp.so.5 (no .a),
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

set -euo pipefail

LLVMORG_VERSION="${1:-22.1.6}"
PREFIX="${PREFIX:-/opt/libomp}"
SRC_DIR="${SRC_DIR:-/tmp/llvm-project}"
BUILD_DIR="${SRC_DIR}/build-openmp"

rm -rf "${SRC_DIR}"
git clone \
    --depth 1 \
    --branch "llvmorg-${LLVMORG_VERSION}" \
    --filter=blob:none \
    --sparse \
    https://github.com/llvm/llvm-project.git \
    "${SRC_DIR}"

cd "${SRC_DIR}"
git sparse-checkout set --no-cone openmp cmake runtimes

cmake \
    -S openmp \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-22 \
    -DCMAKE_CXX_COMPILER=clang++-22 \
    -DLIBOMP_ENABLE_SHARED=OFF \
    -DLIBOMP_OMPD_SUPPORT=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DOPENMP_ENABLE_OMPT_TOOLS=OFF \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}"

cmake --build "${BUILD_DIR}" -j"$(nproc)" --target install

# Sanity: archive present, no .so produced.
test -f "${PREFIX}/lib/libomp.a"
test -f "${PREFIX}/include/omp.h"
# Forbid only libomp.so* (not libompd.so etc.) — we only ship libomp.a.
if find "${PREFIX}/lib" -maxdepth 1 -name 'libomp.so*' | grep -q .; then
    echo "unexpected shared libomp produced under ${PREFIX}" >&2
    find "${PREFIX}/lib" -maxdepth 1 -name 'libomp.so*' >&2
    exit 1
fi

rm -rf "${SRC_DIR}"
