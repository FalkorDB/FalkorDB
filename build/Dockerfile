FROM redis:8.6.2

# Major LLVM channel installed via apt.llvm.org (provides clang-${CLANG_MAJOR},
# libc++-${CLANG_MAJOR}-dev, libpolly-${CLANG_MAJOR}-dev, etc.). apt.llvm.org
# only ships per-major streams; the point release moves whenever upstream
# cuts one. We do NOT pin the point release as a separate ARG — instead the
# libomp build below auto-detects the exact `X.Y.Z` from the installed
# clang-${CLANG_MAJOR} binary, guaranteeing the libomp.a ABI matches the
# clang we just installed even if apt.llvm.org bumps mid-week.
ARG CLANG_MAJOR=22

ENV PATH="/root/.cargo/bin:${PATH}"

RUN apt update -y && \
    apt install -y lsb-release gnupg wget && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh "${CLANG_MAJOR}" && \
    apt update -y

RUN apt install -y git make cmake curl zlib1g zlib1g-dev \
    libpolly-${CLANG_MAJOR}-dev libc++-${CLANG_MAJOR}-dev libc++abi-${CLANG_MAJOR}-dev \
    libzstd-dev python3-venv lcov

# Build LLVM libomp as a static archive (/opt/libomp/lib/libomp.a + headers).
# Isolated in its own layer + auto-versioned from the installed clang so the
# registry buildcache hits whenever libomp.sh is unchanged AND clang's point
# release hasn't moved. Placed before the slower-churning GraphBLAS/LAGraph/
# RediSearch builds so it stays cached across most PRs.
COPY build/libomp.sh /tmp/libomp.sh
# Derive LLVMORG_VERSION from clang-${CLANG_MAJOR} --version (output line 1
# looks like e.g. "Ubuntu clang version 22.1.6"). This makes the libomp.a
# we embed ABI-matched to the clang we ship, with no manual lock-step bump.
RUN LLVMORG_VERSION=$(clang-${CLANG_MAJOR} --version | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1) && \
    test -n "${LLVMORG_VERSION}" || { echo "ERROR: failed to detect LLVMORG_VERSION from clang-${CLANG_MAJOR}" >&2; exit 1; } && \
    test "${LLVMORG_VERSION%%.*}" = "${CLANG_MAJOR}" || { \
        echo "ERROR: clang-${CLANG_MAJOR} reports version ${LLVMORG_VERSION}; major mismatch." >&2; exit 1; \
    } && \
    echo "Building libomp from llvmorg-${LLVMORG_VERSION} (auto-detected from clang-${CLANG_MAJOR})" && \
    CC=clang-${CLANG_MAJOR} CXX=clang++-${CLANG_MAJOR} \
    /tmp/libomp.sh "${LLVMORG_VERSION}" && \
    rm -f /tmp/libomp.sh

# Docker CLI for per-test-class container orchestration in flow tests:
# tests/flow/common.py spawns dedicated containers from the RC image so each
# class gets its own fresh redis with the right load-time module args. The
# CLI talks to the host docker daemon via /var/run/docker.sock (mounted via
# the rust-pr.yml `container.options`). docker.io includes the daemon binary
# we never run — only the client matters here.
RUN apt install -y docker.io

RUN python3 -m venv venv

RUN venv/bin/pip install behave falkordb falkordb-bulk-loader hypothesis pytest pytest-benchmark RLTest

# Build SuiteSparse:GraphBLAS v10.3.1 + LAGraph v1.2.1 with the FalkorDB-
# specific tweaks: GB_control.h patch, vendored PreJIT kernels from the
# FalkorDB C repo, matching cmake flags, and shared OpenMP wiring. Single
# source of truth is graphblas.sh at the repo root — devs invoke it
# directly, CI invokes the same script here. LAGraph is built by the same
# script so both deps stay locked together. Output: /usr/local/lib/libgraphblas.a
# + /data/lagraph_lib/{liblagraph.a,liblagraphx.a,LAGraph.h,LAGraphX.h}.
COPY graphblas.sh /data/graphblas.sh
COPY build/graphblas /data/build/graphblas
RUN cd /data && \
    CC=clang-${CLANG_MAJOR} CXX=clang++-${CLANG_MAJOR} LAGRAPH_INSTALL_DIR=/data/lagraph_lib \
    ./graphblas.sh && \
    rm -rf /data/graphblas.sh /data/build

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN . /data/venv/bin/activate && \
    mkdir -p redisearch && \
    cd redisearch && \
    git clone --recurse-submodules --branch vector-low-level-api --single-branch https://github.com/FalkorDB/RediSearch.git && \
    cd RediSearch && \
    sed -i 's/-Werror//g' deps/VectorSimilarity/src/VecSim/CMakeLists.txt && \
    make build STATIC=1 CLANG=1 CC=clang-${CLANG_MAJOR} CXX=clang++-${CLANG_MAJOR}

RUN cd redisearch && \
    cp -r RediSearch/bin bin  && \
    rm -rf RediSearch

CMD "/bin/bash"
