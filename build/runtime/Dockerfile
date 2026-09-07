ARG BUILD_IMAGE=ghcr.io/falkordb/falkordb-build:latest
ARG BROWSER_TAG=latest
ARG TARGETPLATFORM=linux/amd64
# Redis base image version. Override via `--build-arg REDIS_VERSION=X.Y.Z`
# to ship against a different redis release without editing the Dockerfile.
ARG REDIS_VERSION=8.6.3

FROM $BUILD_IMAGE AS chef
RUN cargo install cargo-chef --locked
WORKDIR /src

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
# CXX defaults to the Debian toolchain's clang++-22. Per-OS toolchain images
# (rhel9/al2023/rhel8/alpine) reuse this exact builder stage and override the
# compiler via `--build-arg CXX=<their clang++>` — keeping one build path /
# one self-containment contract for every platform (see build/Dockerfile.<os>).
ARG CXX=clang++-22
COPY --from=planner /src/recipe.json recipe.json
# cargo-chef stubs out build SCRIPTS, but cargo still has to RESOLVE every
# manifest -- and graph/Cargo.toml carries a path build-dependency on
# ../native-deps. Without the crate present, `cook` fails with
# "failed to read /src/native-deps/Cargo.toml". It is ~140KB of source and its
# own workspace, so copying it here costs nothing and does not disturb the
# dependency layer chef is caching.
COPY native-deps native-deps
COPY deps/native-deps.lock deps/native-deps.lock
RUN CXX="$CXX" cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN CXX="$CXX" cargo build --release && \
    mkdir -p /out && cp target/release/libfalkordb.so /out/falkordb.so && \
    echo "=== verify single-.so contract on libfalkordb.so ===" && \
    deps="$(ldd /out/falkordb.so | awk '{print $1}' | sort -u)" && \
    echo "$deps" && \
    if echo "$deps" | grep -E '^(libomp|libgomp|libgraphblas|liblagraph|liblagraphx|libredisearch|libVectorSimilarity)'; then \
        echo "FAIL: libfalkordb.so has forbidden dynamic dependency"; exit 1; \
    fi && \
    undef="$(nm -D --undefined-only /out/falkordb.so 2>/dev/null | grep -E 'omp_|GOMP_|__kmpc_' || true)" && \
    if [ -n "$undef" ]; then echo "FAIL: OpenMP symbols undefined in .so:"; echo "$undef"; exit 1; fi && \
    echo "OK: libfalkordb.so is self-contained ($(du -h /out/falkordb.so | cut -f1))"

# ---------------------------------------------------------------------------
# Artifact export stage — yields ONLY the built falkordb.so so CI can extract
# the exact .so the runtime image ships via:
#   docker build --target artifact --output type=local,dest=<dir> ...
# This is the single source of truth for the released shared library: it reuses
# the `builder` stage above (same cargo flags, same self-containment contract),
# so the released .so == the .so baked into the runtime image.
#
# FROM scratch keeps the --output tree clean (just the .so, no userland). It is
# placed AFTER `builder` and BEFORE the runtime stages so (a) it never pulls in
# the Debian runtime layers and (b) it is NOT the default build target — the
# default remains the final `full` stage at the bottom of this file.
# ---------------------------------------------------------------------------
FROM scratch AS artifact
COPY --from=builder /out/falkordb.so /falkordb.so

# ---------------------------------------------------------------------------
# Redis builder stage — debian:trixie-slim + minimal build deps.
#
# We build redis-server / redis-cli ourselves on debian:trixie-slim instead
# of inheriting `FROM redis:${REDIS_VERSION}` for two reasons:
#
#   1. The upstream redis image bakes a ~60 MB build-toolchain layer
#      (gcc, g++, cmake, clang, python, rust) that is `apt-get purge`d in
#      the same RUN — purged binaries don't shrink the layer tarball, so
#      that 60 MB ships in every pull. Discarding this entire stage drops
#      the waste.
#
#   2. The upstream image also ships the bundled stack modules
#      (redisearch, rejson, redistimeseries, redisbloom — ~25 MB combined).
#      run.sh only `--loadmodule`s falkordb.so so those modules are dead
#      weight. Building with BUILD_WITH_MODULES=no skips them entirely
#      and eliminates the rust toolchain requirement that pulls them in.
#
# Trade-off: we are now the SBOM/CVE publisher for the redis binaries,
# losing the upstream image's attestation chain. Acceptable for a DB
# vendor distribution.
# ---------------------------------------------------------------------------
FROM debian:trixie-slim AS redis-builder

ARG REDIS_VERSION

ENV DEBIAN_FRONTEND=noninteractive
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates git dpkg-dev gcc g++ libc6-dev libssl-dev make; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
# Clone the immutable upstream tag — same pattern as Dockerfile.asan.
# --depth 1 keeps the fetch small; the tag ref is content-addressed by git
# so no separate SHA pin is required.
RUN set -eux; \
    git clone --depth 1 --branch "${REDIS_VERSION}" https://github.com/redis/redis.git redis-src

WORKDIR /tmp/redis-src
RUN set -eux; \
    # Mirror the upstream image's protected-mode default flip (1 → 0)
    # — docker users supply auth/network controls externally.
    sed -ri 's!^( *createBoolConfig[(]"protected-mode",.*, *)1( *,.*[)],)$!\10\2!' src/config.c; \
    grep -E '^ *createBoolConfig[(]"protected-mode",.*, *0 *,.*[)],$' src/config.c; \
    # Bundled jemalloc page-size tuning to match upstream image.
    gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)"; \
    arch="$(dpkg --print-architecture | awk -F- '{ print $NF }')"; \
    extraJemallocConfigureFlags="--build=$gnuArch"; \
    case "${arch##*-}" in \
        amd64 | i386 | x32) extraJemallocConfigureFlags="$extraJemallocConfigureFlags --with-lg-page=12" ;; \
        *) extraJemallocConfigureFlags="$extraJemallocConfigureFlags --with-lg-page=16" ;; \
    esac; \
    extraJemallocConfigureFlags="$extraJemallocConfigureFlags --with-lg-hugepage=21"; \
    sed -ri 's!cd jemalloc && ./configure !&'"$extraJemallocConfigureFlags"' !' deps/Makefile; \
    export BUILD_TLS=yes BUILD_WITH_MODULES=no; \
    make -j "$(nproc)" all; \
    make PREFIX=/redis-install install; \
    strip /redis-install/bin/redis-server /redis-install/bin/redis-cli; \
    /redis-install/bin/redis-server --version; \
    /redis-install/bin/redis-cli --version

# ---------------------------------------------------------------------------
# Runtime image — debian:trixie-slim, no build tools.
# ---------------------------------------------------------------------------
FROM debian:trixie-slim AS server

# BuildKit auto-ARGs (TARGETPLATFORM, TARGETARCH, ...) must be re-declared in each
# stage that consumes them, else they expand to empty.
ARG TARGETPLATFORM

ENV FALKORDB_HOME=/var/lib/falkordb \
    FALKORDB_DATA_PATH=/var/lib/falkordb/data \
    FALKORDB_BIN_PATH=/var/lib/falkordb/bin \
    FALKORDB_IMPORT_PATH=/var/lib/falkordb/import \
    FALKORDB_TLS_PATH=/var/lib/falkordb/tls

# Runtime-image apt install. After PR #483 GraphBLAS JIT is disabled at module
# init (GxB_JIT_OFF), and libomp is now statically linked into libfalkordb.so
# (see graph/build.rs + build/libomp.sh). The runtime image therefore needs
# NO compiler, NO libomp shared library, and NO LLVM tooling at all — only
# the bare runtime essentials.
#
# Required at runtime:
#   ca-certificates, bash, openssl  : run.sh + gen-certs.sh
#   libssl3                         : redis-server with BUILD_TLS=yes
#
# CVE trim: purge python3.13 + force-remove essential packages not needed at
# runtime (ncurses, perl, login, passwd). With no LLVM in this image any more
# we can run the trim straight through without the load-bearing ordering the
# old layout required.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates bash openssl libssl3; \
    apt-get upgrade -y; \
    # Create the redis user/group BEFORE purging `passwd` (which provides
    # groupadd/useradd). Matches the upstream redis image's uid/gid 999
    # so downstream consumers' bind-mount permissions and `USER redis`
    # directives remain compatible after the base switch.
    groupadd -r -g 999 redis && useradd -r -g redis -u 999 redis; \
    (apt-get purge -y python3.13 python3.13-minimal libpython3.13-minimal libpython3.13-stdlib || true); \
    (dpkg --force-remove-essential --purge ncurses-base ncurses-bin || true); \
    (dpkg --force-remove-essential --purge perl-base || true); \
    (dpkg --force-remove-protected --force-remove-essential --purge login || true); \
    (dpkg --purge passwd || true); \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /var/cache/debconf/*

# Pull only the binaries from the redis-builder stage — no build toolchain,
# no bundled modules, no jemalloc build artefacts.
COPY --from=redis-builder /redis-install/bin/redis-server /usr/local/bin/redis-server
COPY --from=redis-builder /redis-install/bin/redis-cli    /usr/local/bin/redis-cli

RUN mkdir -p ${FALKORDB_DATA_PATH} ${FALKORDB_BIN_PATH} ${FALKORDB_IMPORT_PATH} ${FALKORDB_TLS_PATH} && \
    ln -s ${FALKORDB_DATA_PATH} /data

WORKDIR ${FALKORDB_HOME}

COPY ./build/runtime/run.sh       ${FALKORDB_BIN_PATH}/run.sh
COPY ./build/runtime/gen-certs.sh ${FALKORDB_BIN_PATH}/gen-certs.sh
RUN chmod +x ${FALKORDB_BIN_PATH}/run.sh ${FALKORDB_BIN_PATH}/gen-certs.sh
COPY --from=builder /out/falkordb.so ${FALKORDB_BIN_PATH}/falkordb.so

# Runtime-stage sanity (cheap belt-and-braces; the load-bearing checks are in
# the builder stage where binutils is available). Verify falkordb.so deps and
# that the redis binaries we copied are runnable in this base.
RUN deps="$(ldd ${FALKORDB_BIN_PATH}/falkordb.so | awk '{print $1}' | sort -u)" && \
    echo "$deps" && \
    if echo "$deps" | grep -E '^(libomp|libgomp|libgraphblas|liblagraph|liblagraphx|libredisearch|libVectorSimilarity)'; then \
        echo "FAIL: forbidden dyn dep in final image"; exit 1; \
    fi && \
    redis-server --version && \
    redis-cli --version

ENV ARCH=${TARGETPLATFORM} \
    TLS=0 \
    BROWSER=0 \
    FALKORDB_ARGS="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000"

EXPOSE 6379/tcp

# run.sh is self-contained — it execs `redis-server ...` with the right
# args based on TLS/BROWSER env vars and doesn't forward $@. No CMD
# stanza so it's clear that container args are ignored.
ENTRYPOINT ["/var/lib/falkordb/bin/run.sh"]

FROM falkordb/falkordb-browser:$BROWSER_TAG AS browser
FROM node:26-bookworm-slim AS node

FROM server AS full
ENV FALKORDB_BROWSER_PATH=/var/lib/falkordb/browser \
    BROWSER=1
COPY --from=node /usr/local/bin/node /usr/local/bin/
COPY --from=browser /app ${FALKORDB_BROWSER_PATH}
COPY --from=browser /usr/local/bin/docker-entrypoint.sh ${FALKORDB_BROWSER_PATH}/
EXPOSE 3000/tcp
