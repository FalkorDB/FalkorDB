ARG BUILD_IMAGE=ghcr.io/falkordb/falkordb-build:latest
ARG BROWSER_TAG=latest
ARG TARGETPLATFORM=linux/amd64

# cargo-chef splits dep compilation into a layer cached on Cargo.toml/Cargo.lock,
# so source-only changes skip dep rebuild.
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

FROM redis:8.6.3 AS server

# BuildKit auto-ARGs (TARGETPLATFORM, TARGETARCH, ...) must be re-declared in each
# stage that consumes them, else they expand to empty.
ARG TARGETPLATFORM

ENV FALKORDB_HOME=/var/lib/falkordb \
    FALKORDB_DATA_PATH=/var/lib/falkordb/data \
    FALKORDB_BIN_PATH=/var/lib/falkordb/bin \
    FALKORDB_IMPORT_PATH=/var/lib/falkordb/import \
    FALKORDB_TLS_PATH=/var/lib/falkordb/tls

RUN mkdir -p ${FALKORDB_DATA_PATH} && ln -s ${FALKORDB_DATA_PATH} /data

# Runtime-image apt install. Two distinct phases:
#
# Phase 1 — CVE trim of the redis base image (purge python3.13, force-remove
# ncurses/perl/login/passwd, autoremove orphaned auto deps).
#
# Phase 2 — install the LLVM 22 toolchain LAST, after all autoremoves, so the
# JIT-dep packages can't be purged as collateral. Earlier attempts placed the
# LLVM install before the cleanup and relied on `apt-mark manual` to protect
# libomp-22-dev; that fails because `apt-get autoremove --purge` after the
# dpkg force-removes of essential packages breaks the dep graph and removes
# manually-marked packages anyway (the debian/apt --purge variant ignores the
# manual flag once a package has unmet deps). Installing LLVM after the last
# autoremove sidesteps the issue: there is no subsequent cleanup pass to nuke
# omp.h.
#
# JIT lifecycle: on first matrix-op GraphBLAS generates C source under
# ~/.SuiteSparse/GrBN.M.K/c/ and shells out to `cc` to compile per-kernel
# .so files. Without `cc` + omp.h the JIT call returns GxB_JIT_ERROR and
# matrix.rs's `assert_eq!(info, GrB_SUCCESS)` panics the module on every
# matrix op (MATCH, SET, DELETE, ...).
#
# We install clang-22 + libomp-22-dev (header) + libomp5 (runtime) and point
# `/usr/bin/cc` at clang-22 so the JIT-compiled kernels are built by the
# same compiler that produced our pre-baked libgraphblas.a / liblagraph.a /
# libredisearch-static.a / libfalkordb.so. Heterogeneous toolchains
# (clang-baked statics + gcc-JIT kernels) are technically workable per the
# asan Dockerfile in commit 4215713, but a consistent clang-22 stack
# removes a class of subtle codegen / libomp ABI risks.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates bash && \
    # ---- Phase 1: CVE trim of the redis base ----
    apt-get upgrade -y && \
    (apt-get purge -y python3.13 python3.13-minimal libpython3.13-minimal libpython3.13-stdlib || true) && \
    apt-get autoremove -y && \
    (dpkg --force-remove-essential --purge ncurses-base ncurses-bin || true) && \
    (dpkg --force-remove-essential --purge perl-base || true) && \
    (dpkg --force-remove-protected --force-remove-essential --purge login || true) && \
    (dpkg --purge passwd || true) && \
    apt-get autoremove -y --purge 2>/dev/null || true && \
    # ---- Phase 2: install LLVM toolchain LAST ----
    apt-get install -y --no-install-recommends wget gnupg lsb-release && \
    wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 22 && \
    apt-get install -y --no-install-recommends clang-22 libomp5 libomp-22-dev && \
    ln -sf /usr/bin/clang-22 /usr/bin/cc && \
    apt-get purge -y wget gnupg lsb-release && \
    # No autoremove or autoremove --purge after this — those would reach the
    # JIT deps via the unmet-dep edge case explained above.
    apt-get clean && rm -rf /var/lib/apt/lists/* /llvm.sh

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

ENTRYPOINT ["/var/lib/falkordb/bin/run.sh"]
CMD ["redis-server", "--loadmodule", "/var/lib/falkordb/bin/falkordb.so"]

FROM falkordb/falkordb-browser:$BROWSER_TAG AS browser
FROM node:24-bookworm-slim AS node

FROM server AS full
ENV FALKORDB_BROWSER_PATH=/var/lib/falkordb/browser \
    BROWSER=1
COPY --from=node /usr/local/bin/node /usr/local/bin/
COPY --from=browser /app ${FALKORDB_BROWSER_PATH}
COPY --from=browser /usr/local/bin/docker-entrypoint.sh ${FALKORDB_BROWSER_PATH}/
EXPOSE 3000/tcp
