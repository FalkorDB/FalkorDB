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
COPY --from=planner /src/recipe.json recipe.json
RUN CXX=clang++-22 cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN CXX=clang++-22 cargo build --release && \
    mkdir -p /out && cp target/release/libfalkordb.so /out/falkordb.so

FROM redis:${REDIS_VERSION} AS server

# BuildKit auto-ARGs (TARGETPLATFORM, TARGETARCH, ...) must be re-declared in each
# stage that consumes them, else they expand to empty.
ARG TARGETPLATFORM

ENV FALKORDB_HOME=/var/lib/falkordb \
    FALKORDB_DATA_PATH=/var/lib/falkordb/data \
    FALKORDB_BIN_PATH=/var/lib/falkordb/bin \
    FALKORDB_IMPORT_PATH=/var/lib/falkordb/import \
    FALKORDB_TLS_PATH=/var/lib/falkordb/tls

RUN mkdir -p ${FALKORDB_DATA_PATH} && ln -s ${FALKORDB_DATA_PATH} /data

# Runtime-image apt install. GraphBLAS's JIT is permanently disabled at
# module init (see graph/src/graph/graphblas/matrix.rs `GxB_JIT_OFF`, #452)
# so the runtime image no longer needs clang or libomp-22-dev — no kernel
# compilation happens at runtime. libomp.so.5 is still required by
# libfalkordb.so and is copied from the builder stage further below.
#
# PreJIT kernel curation is the follow-up that recovers the performance
# otherwise lost to generic-kernel fallbacks (see project tracker).
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates bash openssl && \
    apt-get upgrade -y && \
    # CVE trim — packages not needed at runtime by redis-server / falkordb.so.
    (apt-get purge -y python3.13 python3.13-minimal libpython3.13-minimal libpython3.13-stdlib || true) && \
    (dpkg --force-remove-essential --purge ncurses-base ncurses-bin || true) && \
    (dpkg --force-remove-essential --purge perl-base || true) && \
    (dpkg --force-remove-protected --force-remove-essential --purge login || true) && \
    (dpkg --purge passwd || true) && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR ${FALKORDB_HOME}

COPY ./build/runtime/run.sh       ${FALKORDB_BIN_PATH}/run.sh
COPY ./build/runtime/gen-certs.sh ${FALKORDB_BIN_PATH}/gen-certs.sh
RUN chmod +x ${FALKORDB_BIN_PATH}/run.sh ${FALKORDB_BIN_PATH}/gen-certs.sh
COPY --from=builder /out/falkordb.so ${FALKORDB_BIN_PATH}/falkordb.so

# libomp.so.5 (LLVM OpenMP runtime) is linked by libfalkordb.so but absent from
# the redis base. TARGETARCH selects the correct multiarch path so the ABI
# matches what the binary was compiled against on both linux/amd64 and linux/arm64.
ARG TARGETARCH
RUN --mount=from=builder,target=/builder,readonly \
    case "${TARGETARCH}" in \
        amd64) LIBOMP=/builder/usr/lib/x86_64-linux-gnu/libomp.so.5 ;; \
        arm64) LIBOMP=/builder/usr/lib/aarch64-linux-gnu/libomp.so.5 ;; \
        *) echo "unsupported TARGETARCH=${TARGETARCH}"; exit 1 ;; \
    esac && \
    cp "${LIBOMP}" /usr/local/lib/libomp.so.5 && \
    ldconfig

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
FROM node:24-bookworm-slim AS node

FROM server AS full
ENV FALKORDB_BROWSER_PATH=/var/lib/falkordb/browser \
    BROWSER=1
COPY --from=node /usr/local/bin/node /usr/local/bin/
COPY --from=browser /app ${FALKORDB_BROWSER_PATH}
COPY --from=browser /usr/local/bin/docker-entrypoint.sh ${FALKORDB_BROWSER_PATH}/
EXPOSE 3000/tcp
