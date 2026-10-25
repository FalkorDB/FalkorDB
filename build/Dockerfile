# Toolchain image (ghcr.io/falkordb/falkordb-build) — internal CI image used
# to compile libfalkordb.so, GraphBLAS, LAGraph, and to run flow / bench
# tests. NOT shipped to users.
#
# We build redis-server / redis-cli and valkey-server / valkey-cli from git
# clones of the pinned tags (build/build-server.sh) rather than inheriting
# `FROM redis:${REDIS_VERSION}`. This:
#   * gives us the SAME server binaries across the toolchain image, the
#     release runtime images (build/runtime/Dockerfile{,.rhel9,.amazonlinux2023})
#     and the coverage runtime image (build/runtime/Dockerfile.coverage) — all
#     use the same git tags, same flags, same protected-mode + jemalloc tweaks.
#   * keeps consistency with build/runtime/Dockerfile.asan which already
#     builds its own (ASAN-instrumented) server from git.
#   * removes the implicit dependency on upstream's redis: image metadata
#     (entrypoint, gosu, redis user) which we override anyway.
#
# The versions match every other Dockerfile that builds a server;
# build/check-server-pins.sh enforces it in CI.
ARG REDIS_VERSION=8.10.2
ARG VALKEY_VERSION=9.1.2

FROM debian:trixie-slim AS server-builder-base
ENV DEBIAN_FRONTEND=noninteractive
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates git gcc g++ libc6-dev libssl-dev make; \
    rm -rf /var/lib/apt/lists/*
COPY ./build/build-server.sh /usr/local/bin/build-server

FROM server-builder-base AS redis-builder
ARG REDIS_VERSION
RUN build-server redis "${REDIS_VERSION}" /server-install

# This image ships both servers side by side, so valkey's redis-* aliases are
# left out: redis-server here is the real Redis.
FROM server-builder-base AS valkey-builder
ARG VALKEY_VERSION
RUN REDIS_ALIASES=no build-server valkey "${VALKEY_VERSION}" /server-install

# ---------------------------------------------------------------------------
# deps-base: everything the native dependency builds need, and nothing else.
#
# The stages below branch from here, so ONLY put things here that GraphBLAS,
# LAGraph or RediSearch actually need to compile. Convenience tooling for the
# test harness (docker CLI, lcov, the python venv) lives in the final stage,
# BELOW the branch point, so adding a pytest dependency can never invalidate a
# 12-minute GraphBLAS build.
# ---------------------------------------------------------------------------
FROM debian:trixie-slim AS deps-base

# Major LLVM channel installed via apt.llvm.org (provides clang-${CLANG_MAJOR},
# libc++-${CLANG_MAJOR}-dev, libpolly-${CLANG_MAJOR}-dev, etc.). apt.llvm.org
# only ships per-major streams; the point release moves whenever upstream
# cuts one. We do NOT pin the point release as a separate ARG — instead the
# libomp build below auto-detects the exact `X.Y.Z` from the installed
# clang-${CLANG_MAJOR} binary, guaranteeing the libomp.a ABI matches the
# clang we just installed even if apt.llvm.org bumps mid-week.
ARG CLANG_MAJOR=23

ENV PATH="/root/.cargo/bin:${PATH}"
ENV DEBIAN_FRONTEND=noninteractive

# Base runtime libs that previously came with the `redis:` base image
# (ca-certificates for git/curl HTTPS, libssl3 for the redis-server binary
# copied into the final stage).
RUN apt update -y && \
    apt install -y --no-install-recommends \
        ca-certificates libssl3 openssl && \
    apt install -y lsb-release gnupg wget && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh "${CLANG_MAJOR}" && \
    apt update -y

# Toolchain + headers the three deps compile against.
# libssl-dev: RediSearch 8.6's coordinator/hiredis_ssl link against
# libssl/libcrypto, so it is a genuine *build* input and belongs here.
RUN apt install -y git make cmake curl zlib1g zlib1g-dev \
    libpolly-${CLANG_MAJOR}-dev libc++-${CLANG_MAJOR}-dev libc++abi-${CLANG_MAJOR}-dev \
    libzstd-dev libssl-dev

# Build LLVM libomp as a static archive (/opt/libomp/lib/libomp.a + headers).
# Isolated in its own layer + auto-versioned from the installed clang so the
# registry buildcache hits whenever libomp.sh is unchanged AND clang's point
# release hasn't moved.
COPY build/libomp.sh /tmp/libomp.sh
# Derive LLVMORG_VERSION from clang-${CLANG_MAJOR} --version (output line 1
# looks like e.g. "Ubuntu clang version 23.1.2"). This makes the libomp.a
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

# native-deps is a Rust program, so cargo must exist BEFORE the dep stages.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# CC/CXX/LIBOMP_PREFIX are cache-key INPUTS for native-deps, not mere build
# preferences: the key a `cargo build` computes in the final stage must equal
# the key each dep stage published under, or every prebuilt artifact misses and
# gets rebuilt from scratch. Pin them as ENV once, here, so every stage that
# inherits from deps-base agrees. (Passing `CC=clang-23` inline on individual
# RUN lines is NOT enough — cmake would probe `cc` elsewhere.)
ENV CC=clang-${CLANG_MAJOR}
ENV CXX=clang++-${CLANG_MAJOR}
ENV LIBOMP_PREFIX=/opt/libomp

# Where the dep stages publish their artifacts, so the final stage can COPY
# them wholesale out of a known root.
ENV FALKORDB_DEPS_CACHE=/opt/falkordb-deps

WORKDIR /src

# ---------------------------------------------------------------------------
# graphblas-build: SuiteSparse:GraphBLAS + LAGraph.
#
# COPYs ONLY its own inputs, so a RediSearch pin bump cannot invalidate it (and
# vice versa). build/graphblas carries GB_control.patch and the vendored PreJIT
# kernels; both are cache-key inputs, so a re-harvest produces a new key rather
# than a silently stale archive.
# ---------------------------------------------------------------------------
FROM deps-base AS graphblas-build
COPY native-deps           /src/native-deps
COPY deps/native-deps.lock /src/deps/native-deps.lock
COPY build/graphblas       /src/build/graphblas
COPY deps/GraphBLAS        /src/deps/GraphBLAS
COPY deps/LAGraph          /src/deps/LAGraph
RUN cargo run --locked --manifest-path native-deps/Cargo.toml -- graphblas lagraph

# ---------------------------------------------------------------------------
# redisearch-build: the RediSearch 8.6 fork as an embeddable static library.
#
# The deps/RediSearch submodule is the single source of truth for WHICH commit,
# so this layer's cache key is the submodule's content: it turns over exactly
# when the pin moves. Requires `submodules: recursive` on actions/checkout;
# deps/RediSearch/bin is .dockerignore'd so a host-built (wrong-arch) output can
# never come along.
#
# No python here on purpose: build.sh only touches python inside
# run_python_tests(), which early-returns unless RUN_PYTEST=1.
# ---------------------------------------------------------------------------
FROM deps-base AS redisearch-build
COPY native-deps           /src/native-deps
COPY deps/native-deps.lock /src/deps/native-deps.lock
COPY deps/RediSearch       /src/deps/RediSearch
RUN cargo run --locked --manifest-path native-deps/Cargo.toml -- redisearch

# ---------------------------------------------------------------------------
# Final toolchain image.
# ---------------------------------------------------------------------------
FROM deps-base

ARG REDIS_VERSION
ARG VALKEY_VERSION

# Install the self-built servers at the same path the upstream `redis:` image
# used (/usr/local/bin), so tests/common.py ("/usr/local/bin/redis-server",
# or valkey-server with FALKORDB_SERVER=valkey) and RLTest's PATH lookup
# pick them up transparently. Directory COPYs, so the sentinel/check-*
# symlinks come along.
COPY --from=redis-builder  /server-install/ /usr/local/
COPY --from=valkey-builder /server-install/ /usr/local/
RUN redis-server --version | grep -F " v=${REDIS_VERSION} " && redis-cli --version && \
    valkey-server --version | grep -F " v=${VALKEY_VERSION} " && valkey-cli --version

# ---- test-harness convenience, BELOW the dep branch point ----
# Everything from here down can churn freely without rebuilding a dep.

# Docker CLI for per-test-class container orchestration in flow tests:
# tests/flow/common.py spawns dedicated containers from the RC image so each
# class gets its own fresh redis with the right load-time module args. The
# CLI talks to the host docker daemon via /var/run/docker.sock (mounted via
# the rust-pr.yml `container.options`). docker.io includes the daemon binary
# we never run — only the client matters here.
RUN apt install -y docker.io lcov python3-venv

# The upstream `redis:` base used to set WORKDIR /data implicitly; we no
# longer inherit that, so set it explicitly here. The venv install
# (`python3 -m venv venv`) assumes /data is the cwd so the venv lands at
# /data/venv (referenced absolutely elsewhere as `/data/venv/bin/activate`).
WORKDIR /data

RUN python3 -m venv venv

# `redis<8.1` is a pin against falkordb-py, not a preference: redis-py 8.1.0
# puts `himport_registry` into `ConnectionPool.connection_kwargs`, and
# falkordb-py 1.6.2's async `Is_Cluster` forwards those kwargs verbatim to the
# *sync* `Redis(**kwargs)`, which rejects it —
# `TypeError: Redis.__init__() got an unexpected keyword argument
# 'himport_registry'`. Every flow test that builds its own pool
# (test_cache, test_concurrent_query, test_timeout, test_slowlog,
# test_index_create, test_query_mem_limit, test_pending_queries_limit) dies on
# it. The pin only bites when this image is rebuilt, which is why it surfaced
# on a PR touching build/graphblas rather than on trunk. Drop it once
# falkordb-py releases a fix.
RUN venv/bin/pip install behave falkordb 'redis<8.1' falkordb-bulk-loader hypothesis pytest RLTest uv

# ---- prebuilt native deps ----
# Both stages publish into the same root, so the two COPYs compose into one
# tree. FALKORDB_NATIVE_DEPS_PREBUILT marks it READ-ONLY: `cargo build` resolves
# against it and, because CC/CXX/LIBOMP_PREFIX above match what the dep stages
# saw, computes the identical key and hits rather than rebuilds.
COPY --from=graphblas-build  /opt/falkordb-deps /opt/falkordb-deps
COPY --from=redisearch-build /opt/falkordb-deps /opt/falkordb-deps
ENV FALKORDB_NATIVE_DEPS_PREBUILT=/opt/falkordb-deps

CMD ["/bin/bash"]
