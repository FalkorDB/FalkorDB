FROM ubuntu:26.04

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
RUN /data/venv/bin/pip install setuptools behave falkordb falkordb-bulk-loader hypothesis pytest RLTest

# Install Rust (needed before running scripts)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Set working directory for build scripts
WORKDIR /data

# Prebuild the three native deps so a fresh container can `cargo build` without
# a 12-minute wait. native-deps owns the recipes and caches the results under
# $HOME/.cache/falkordb/native-deps, keyed on the submodule revisions, the
# GB_control patch, the vendored PreJIT kernels, $CC/$CXX and the OpenMP
# flavour. CC/CXX are ENV rather than per-RUN because they are key inputs: the
# key `cargo build` computes inside the container must match what is baked here.
ENV CC=clang-22
ENV CXX=clang++-22
COPY native-deps           /data/native-deps
COPY deps/native-deps.lock /data/deps/native-deps.lock
COPY build/graphblas       /data/build/graphblas
COPY deps/GraphBLAS        /data/deps/GraphBLAS
COPY deps/LAGraph          /data/deps/LAGraph
COPY deps/RediSearch       /data/deps/RediSearch
RUN cd /data && \
    cargo run --offline --manifest-path native-deps/Cargo.toml -- ensure --all && \
    rm -rf /data/native-deps /data/build /data/deps

# Mark the populated cache read-only for the workspace build. The developer's
# bind-mounted checkout resolves against it, so an untouched dep is never
# rebuilt inside the container.
ENV FALKORDB_NATIVE_DEPS_PREBUILT=/root/.cache/falkordb/native-deps

WORKDIR /workspace

CMD ["/bin/bash"]
