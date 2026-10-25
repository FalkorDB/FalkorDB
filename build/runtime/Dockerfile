ARG BUILD_IMAGE=ghcr.io/falkordb/falkordb-build:latest
ARG BROWSER_TAG=latest
ARG TARGETPLATFORM=linux/amd64
# Which server the image ships FalkorDB on: `redis` (default) or `valkey`.
# Selects the `server-${SERVER}` stage below; the other server's builder stage
# is never built.
ARG SERVER=redis
# Server versions. Override via `--build-arg REDIS_VERSION=X.Y.Z` /
# `--build-arg VALKEY_VERSION=X.Y.Z` to ship against a different release
# without editing the Dockerfile. Every Dockerfile that builds a server pins
# the same versions; build/check-server-pins.sh enforces it in CI.
ARG REDIS_VERSION=8.10.2
ARG VALKEY_VERSION=9.1.2

FROM $BUILD_IMAGE AS chef
RUN cargo install cargo-chef --locked
WORKDIR /src

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
# CXX defaults to the Debian toolchain's clang++-23. Per-OS toolchain images
# (rhel9/al2023/rhel8/alpine) reuse this exact builder stage and override the
# compiler via `--build-arg CXX=<their clang++>` — keeping one build path /
# one self-containment contract for every platform (see build/Dockerfile.<os>).
ARG CXX=clang++-23
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
    unresolved="$(ldd -r /out/falkordb.so 2>&1 | grep -E 'undefined symbol|not found' || true)" && \
    if [ -n "$unresolved" ]; then echo "FAIL: libraries or symbols this OS cannot resolve (the module would not load):"; echo "$unresolved"; exit 1; fi && \
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
# Server builder stages — debian:trixie-slim + minimal build deps.
#
# We build the server binaries ourselves (build/build-server.sh, shared with
# every other image that ships a server) instead of inheriting
# `FROM redis:${REDIS_VERSION}` / `FROM valkey/valkey`, for two reasons:
#
#   1. The upstream redis image bakes a ~60 MB build-toolchain layer
#      (gcc, g++, cmake, clang, python, rust) that is `apt-get purge`d in
#      the same RUN — purged binaries don't shrink the layer tarball, so
#      that 60 MB ships in every pull. Discarding these stages drops
#      the waste.
#
#   2. The upstream redis image also ships the bundled stack modules
#      (redisearch, rejson, redistimeseries, redisbloom — ~25 MB combined).
#      run.sh only `--loadmodule`s falkordb.so so those modules are dead
#      weight; a plain build of the tag carries none of them.
#
# Trade-off: we are now the SBOM/CVE publisher for the server binaries,
# losing the upstream image's attestation chain. Acceptable for a DB
# vendor distribution.
# ---------------------------------------------------------------------------
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

FROM server-builder-base AS valkey-builder
ARG VALKEY_VERSION
RUN build-server valkey "${VALKEY_VERSION}" /server-install

# ---------------------------------------------------------------------------
# Runtime image — debian:trixie-slim, no build tools.
# ---------------------------------------------------------------------------
FROM debian:trixie-slim AS server-base

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
#   libssl3                         : the server, built with BUILD_TLS=yes
#   gosu                            : run.sh drops root to `falkordb`; the
#                                     browser entrypoint drops to nextjs
#
# CVE trim: purge python3.13 + force-remove essential packages not needed at
# runtime (ncurses, perl, login, passwd, util-linux). With no LLVM in this image any more
# we can run the trim straight through without the load-bearing ordering the
# old layout required.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates bash openssl libssl3 gosu; \
    apt-get upgrade -y; \
    # Create the falkordb user/group, which run.sh runs the server as,
    # BEFORE purging `passwd` (which provides groupadd/useradd). uid/gid 999
    # is the upstream redis (and valkey) image's, so bind-mount permissions
    # carry over.
    groupadd -r -g 999 falkordb && useradd -r -g falkordb -u 999 falkordb; \
    # The `full` stage below runs the bundled browser as nextjs:nodejs (1001),
    # the uid/gid baked into the falkordb-browser image. useradd/groupadd only
    # exist until `passwd` is purged a few lines down, so the account has to be
    # created here even though the browser-less `server` image never uses it.
    groupadd -g 1001 nodejs && useradd -r -u 1001 -g nodejs nextjs; \
    (apt-get purge -y python3.13 python3.13-minimal libpython3.13-minimal libpython3.13-stdlib || true); \
    (dpkg --force-remove-essential --purge ncurses-base ncurses-bin || true); \
    (dpkg --force-remove-essential --purge perl-base || true); \
    (dpkg --force-remove-protected --force-remove-essential --purge login || true); \
    (dpkg --purge passwd || true); \
    # util-linux, mount and bsdutils ship the setuid mount/umount/su plus
    # nsenter/unshare, and between them carry most of the image's HIGH
    # findings that Debian has not fixed yet. Nothing here runs them, and no
    # other installed package depends on them or on their five libraries.
    # No `|| true`: if a new dependent appears, fail the build rather than
    # silently ship them — the Trivy gate ignores unfixed findings, so
    # nothing else would notice.
    dpkg --force-remove-essential --purge \
        util-linux mount bsdutils \
        libmount1 libblkid1 libuuid1 libsmartcols1 liblastlog2-2; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /var/cache/debconf/*

# root:falkordb 0775: `falkordb` writes through the group (after run.sh drops
# to it, or under --user 999), and root still owns them, so a root process
# with no capabilities (--cap-drop ALL, where run.sh cannot drop) can still
# write. The official redis image's user:user ownership cannot SAVE in that
# case.
RUN mkdir -p ${FALKORDB_DATA_PATH} ${FALKORDB_BIN_PATH} ${FALKORDB_IMPORT_PATH} ${FALKORDB_TLS_PATH} && \
    chown root:falkordb ${FALKORDB_DATA_PATH} ${FALKORDB_TLS_PATH} && \
    chmod 0775 ${FALKORDB_DATA_PATH} ${FALKORDB_TLS_PATH} && \
    ln -s ${FALKORDB_DATA_PATH} /data

# Pull only the server tree from the builder stage — no build toolchain, no
# bundled modules, no jemalloc build artefacts. build-server.sh also lays down
# the <server>-sentinel / -check-rdb / -check-aof entry points (and on valkey
# the redis-* aliases, so scripts and healthchecks written for the redis image
# keep working) as symlinks: the directory COPY keeps them, at zero size cost.
# FALKORDB_SERVER is what run.sh execs (`${FALKORDB_SERVER}-server`).
FROM server-base AS server-redis
ARG REDIS_VERSION
ENV FALKORDB_SERVER=redis \
    FALKORDB_SERVER_VERSION=${REDIS_VERSION}
COPY --from=redis-builder /server-install/ /usr/local/

FROM server-base AS server-valkey
ARG VALKEY_VERSION
ENV FALKORDB_SERVER=valkey \
    FALKORDB_SERVER_VERSION=${VALKEY_VERSION}
COPY --from=valkey-builder /server-install/ /usr/local/

FROM server-${SERVER} AS server

# BuildKit auto-ARGs (TARGETPLATFORM, TARGETARCH, ...) must be re-declared in each
# stage that consumes them, else they expand to empty.
ARG TARGETPLATFORM

LABEL com.falkordb.server.name="${FALKORDB_SERVER}" \
      com.falkordb.server.version="${FALKORDB_SERVER_VERSION}"

WORKDIR ${FALKORDB_HOME}

COPY ./build/runtime/run.sh       ${FALKORDB_BIN_PATH}/run.sh
COPY ./build/runtime/gen-certs.sh ${FALKORDB_BIN_PATH}/gen-certs.sh
RUN chmod +x ${FALKORDB_BIN_PATH}/run.sh ${FALKORDB_BIN_PATH}/gen-certs.sh
COPY --from=builder /out/falkordb.so ${FALKORDB_BIN_PATH}/falkordb.so

# Runtime-stage sanity (cheap belt-and-braces; the load-bearing checks are in
# the builder stage where binutils is available). Verify falkordb.so deps,
# that the shipped server is the one the labels name, and that every entry
# point this image is expected to ship is present and runnable in this base —
# the sentinel/check-* symlinks are covered here so a future COPY-only
# refactor cannot silently drop them again (issue #2827).
RUN deps="$(ldd ${FALKORDB_BIN_PATH}/falkordb.so | awk '{print $1}' | sort -u)" && \
    echo "$deps" && \
    if echo "$deps" | grep -E '^(libomp|libgomp|libgraphblas|liblagraph|liblagraphx|libredisearch|libVectorSimilarity)'; then \
        echo "FAIL: forbidden dyn dep in final image"; exit 1; \
    fi && \
    "${FALKORDB_SERVER}-server" --version | grep -F " v=${FALKORDB_SERVER_VERSION} " && \
    for name in "${FALKORDB_SERVER}" redis; do \
        for tool in server cli sentinel check-aof; do \
            "${name}-${tool}" --version || exit 1; \
        done; \
        # valkey-check-rdb reads --version only after a file argument, and
        # otherwise checks a file named "--version": either reply proves the
        # entry point runs in check-rdb mode.
        "${name}-check-rdb" --version 2>&1 | grep -E 'check-rdb|Checking RDB file' || exit 1; \
    done

ENV ARCH=${TARGETPLATFORM} \
    TLS=0 \
    BROWSER=0 \
    FALKORDB_ARGS="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000"

EXPOSE 6379/tcp

# run.sh is self-contained — it execs `${FALKORDB_SERVER}-server ...` with the
# right args based on TLS/BROWSER env vars and doesn't forward $@. No CMD
# stanza so it's clear that container args are ignored.
ENTRYPOINT ["/var/lib/falkordb/bin/run.sh"]

FROM falkordb/falkordb-browser:$BROWSER_TAG AS browser
FROM node:26-bookworm-slim AS node

FROM server AS full
ENV FALKORDB_BROWSER_PATH=/var/lib/falkordb/browser \
    BROWSER=1

# The falkordb-browser image is Alpine-based; only its /app tree and entrypoint
# are copied here, so this stage must provide the userland they expect:
#   * node(1) from node:26 links against libatomic.so.1 and libstdc++.so.6,
#     neither of which is in debian:trixie-slim.
#   * docker-entrypoint.sh drops privileges with su-exec(1), an Alpine tool.
#     Debian's equivalent is gosu (installed in `server`), symlinked under
#     the expected name.
# Without these the browser never starts (see issue #2750).
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends libatomic1 libstdc++6; \
    ln -sf "$(command -v gosu)" /usr/local/bin/su-exec; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /var/cache/debconf/*

COPY --from=node /usr/local/bin/node /usr/local/bin/
COPY --from=browser --chown=nextjs:nodejs /app ${FALKORDB_BROWSER_PATH}
COPY --from=browser /usr/local/bin/docker-entrypoint.sh ${FALKORDB_BROWSER_PATH}/

# docker-entrypoint.sh stores the browser's credentials under a hardcoded
# /app/.data, which is outside the copied browser tree — pre-create it owned by
# nextjs so the entrypoint's writability check passes.
RUN mkdir -p /app/.data && chown -R nextjs:nodejs /app/.data && \
    su-exec nextjs:nodejs node --version

EXPOSE 3000/tcp
