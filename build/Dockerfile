FROM redis:8.6.2

# Pinned LLVM source release used to build libomp.a from source (see
# build/libomp.sh). apt.llvm.org's libomp-22-dev ships only the shared
# library; we need the static archive to embed in libfalkordb.so so the
# final runtime image has no libomp dynamic dependency. Bump in lock-step
# with the clang-22 apt package whose ABI must match.
ARG LLVMORG_VERSION=22.1.6

ENV PATH="/root/.cargo/bin:${PATH}"

RUN apt update -y && \
    apt install -y lsb-release gnupg wget && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh "${LLVMORG_VERSION%%.*}" && \
    apt update -y

RUN apt install -y git make cmake curl zlib1g zlib1g-dev libpolly-22-dev libc++-22-dev libc++abi-22-dev libzstd-dev python3-venv lcov

# Build LLVM libomp as a static archive (/opt/libomp/lib/libomp.a + headers).
# Isolated in its own layer + pinned via ARG so the registry buildcache hits
# whenever LLVMORG_VERSION and libomp.sh are unchanged. Placed before the
# slower-churning GraphBLAS/LAGraph/RediSearch builds so it stays cached
# across most PRs.
COPY build/libomp.sh /tmp/libomp.sh
RUN /tmp/libomp.sh "${LLVMORG_VERSION}" && rm -f /tmp/libomp.sh

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
    CC=clang-22 CXX=clang++-22 LAGRAPH_INSTALL_DIR=/data/lagraph_lib \
    ./graphblas.sh && \
    rm -rf /data/graphblas.sh /data/build

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN . /data/venv/bin/activate && \
    mkdir -p redisearch && \
    cd redisearch && \
    git clone --recurse-submodules --branch vector-low-level-api --single-branch https://github.com/FalkorDB/RediSearch.git && \
    cd RediSearch && \
    sed -i 's/-Werror//g' deps/VectorSimilarity/src/VecSim/CMakeLists.txt && \
    make build STATIC=1 CLANG=1 CC=clang-22 CXX=clang++-22

RUN cd redisearch && \
    cp -r RediSearch/bin bin  && \
    rm -rf RediSearch

CMD "/bin/bash"
