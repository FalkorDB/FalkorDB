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

# Runtime-image apt install. After PR #483 GraphBLAS JIT is disabled at module
# init (GxB_JIT_OFF), and libomp is now statically linked into libfalkordb.so
# (see graph/build.rs + build/libomp.sh). The runtime image therefore needs
# NO compiler, NO libomp shared library, and NO LLVM tooling at all — only
# the bare runtime essentials.
#
# Previous revisions of this file installed clang-22 + libomp-22-dev + libomp5
# + a host of dpkg force-removes to survive the LLVM install. All of that is
# now dead code and has been removed. See git history if you need to resurrect
# it (e.g. to re-enable runtime JIT).
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates bash openssl && \
    apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR ${FALKORDB_HOME}

COPY ./build/runtime/run.sh       ${FALKORDB_BIN_PATH}/run.sh
COPY ./build/runtime/gen-certs.sh ${FALKORDB_BIN_PATH}/gen-certs.sh
RUN chmod +x ${FALKORDB_BIN_PATH}/run.sh ${FALKORDB_BIN_PATH}/gen-certs.sh
COPY --from=builder /out/falkordb.so ${FALKORDB_BIN_PATH}/falkordb.so

# Runtime-stage sanity (cheap belt-and-braces; the load-bearing checks are in
# the builder stage where binutils is available). Just `ldd` here.
RUN deps="$(ldd ${FALKORDB_BIN_PATH}/falkordb.so | awk '{print $1}' | sort -u)" && \
    echo "$deps" && \
    if echo "$deps" | grep -E '^(libomp|libgomp|libgraphblas|liblagraph|liblagraphx|libredisearch|libVectorSimilarity)'; then \
        echo "FAIL: forbidden dyn dep in final image"; exit 1; \
    fi

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
