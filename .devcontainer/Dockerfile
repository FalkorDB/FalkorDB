FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CXX=clang++-23

# Install LLVM 21 and build dependencies
RUN apt update -y && \
    apt install -y software-properties-common wget && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 23 && \
    apt update -y

# Install development dependencies including Redis
RUN apt install -y \
    git \
    make \
    cmake \
    curl \
    zlib1g \
    zlib1g-dev \
    libpolly-23-dev \
    libomp-23-dev \
    libc++-23-dev \
    libc++abi-23-dev \
    libzstd-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    gcc \
    lcov \
    clang-23 \
    llvm-23 \
    llvm-23-dev \
    llvm-23-tools \
    lld-23 \
    redis-server \
    sudo

# Symlink redis-server to /usr/local/bin for tests that expect it there
RUN ln -sf /usr/bin/redis-server /usr/local/bin/redis-server

# Create clang/clang++ symlinks so scripts that reference unversioned names work
RUN ln -sf /usr/bin/clang-23 /usr/local/bin/clang && \
    ln -sf /usr/bin/clang++-23 /usr/local/bin/clang++

# Create Python virtual environment for testing
RUN python3 -m venv /data/venv

# Install Python test dependencies (setuptools needed for pkg_resources)
RUN /data/venv/bin/pip install setuptools behave falkordb falkordb-bulk-loader hypothesis pytest RLTest

# Install Rust (needed before running scripts)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Set working directory for build scripts
WORKDIR /data

# Copy build scripts, and the RediSearch submodule they build (git owns the pin;
# redisearch.sh only builds what is checked out, so the source must be present)
COPY graphblas.sh redisearch.sh /data/
COPY deps/RediSearch /data/deps/RediSearch

# Build and install GraphBLAS using project script
RUN chmod +x graphblas.sh && \
    CC=clang-23 CXX=clang++-23 ./graphblas.sh

# Build and install RediSearch using project script
# Build in /data so output lands at /data/deps/RediSearch/bin/, which is one of
# graph/build.rs's absolute fallbacks -- no symlink needed.
# RediSearch's redisearch_rs build scripts find their source root by walking up
# for a `.git` entry (build_utils::git_root; existence check only, never invokes
# git). All git metadata is .dockerignore'd, so create an empty marker -- see
# .dockerignore for why this is a marker rather than the real gitlink.
RUN mkdir -p /data/deps/RediSearch/.git && \
    chmod +x redisearch.sh && \
    ./redisearch.sh

WORKDIR /workspace

CMD ["/bin/bash"]
