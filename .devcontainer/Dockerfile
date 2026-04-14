FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CXX=clang++-22

# Install LLVM 21 and build dependencies
RUN apt update -y && \
    apt install -y software-properties-common wget && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 22 && \
    apt update -y

# Install development dependencies including Redis
RUN apt install -y \
    git \
    make \
    cmake \
    curl \
    zlib1g \
    zlib1g-dev \
    libpolly-22-dev \
    libomp-22-dev \
    libc++-22-dev \
    libc++abi-22-dev \
    libzstd-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    gcc \
    lcov \
    clang-22 \
    llvm-22 \
    llvm-22-dev \
    llvm-22-tools \
    lld-22 \
    redis-server \
    sudo

# Symlink redis-server to /usr/local/bin for tests that expect it there
RUN ln -sf /usr/bin/redis-server /usr/local/bin/redis-server

# Create clang/clang++ symlinks so scripts that reference unversioned names work
RUN ln -sf /usr/bin/clang-22 /usr/local/bin/clang && \
    ln -sf /usr/bin/clang++-22 /usr/local/bin/clang++

# Create Python virtual environment for testing
RUN python3 -m venv /data/venv

# Install Python test dependencies (setuptools needed for pkg_resources)
RUN /data/venv/bin/pip install setuptools behave falkordb falkordb-bulk-loader hypothesis pytest pytest-benchmark RLTest

# Install Rust (needed before running scripts)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Set working directory for build scripts
WORKDIR /data

# Copy build scripts
COPY graphblas.sh redisearch.sh /data/

# Build and install GraphBLAS using project script
RUN chmod +x graphblas.sh && \
    CC=clang-22 CXX=clang++-22 ./graphblas.sh

# Build and install RediSearch using project script
# Build in /data so output lands at /data/redisearch/RediSearch/
# Symlink bin/ so build.rs can find it at /data/redisearch/bin/
RUN chmod +x redisearch.sh && \
    ./redisearch.sh && \
    ln -sf /data/redisearch/RediSearch/bin /data/redisearch/bin

WORKDIR /workspace

CMD ["/bin/bash"]
