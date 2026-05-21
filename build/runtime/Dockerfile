ARG BUILD_IMAGE=ghcr.io/falkordb/falkordb-build:latest
ARG BROWSER_TAG=latest
ARG TARGETPLATFORM=linux/amd64

# cargo-chef splits dep compilation into a layer cached on
# Cargo.toml/Cargo.lock, so source-only changes skip dep rebuild.
# Confirmed not the trigger for the index/edge-SET regressions:
# dropping it (commit c51c730) didn't change the failing test set.
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

# Runtime-image apt install. GraphBLAS JIT shells out to `cc` and #include
# <omp.h> at runtime, so we need clang-22 + libomp-22-dev present in the
# final image. The ordering below is load-bearing:
#
#   1. CVE trim of python3.13 MUST happen BEFORE the LLVM install. Otherwise
#      `apt-get purge -y python3.13 ...` cascades through reverse-deps
#      (llvm-22-tools → python3, etc.) and silently removes the just-installed
#      libomp-22-dev along with clang-tools-22 / liblldb-22 / libpython3.13.
#      Verified: build log on 618a276 (LLVM-first) showed libomp-22-dev
#      installed at 35.84 and then purged at 37.27 by the python cascade.
#
#   2. llvm.sh MUST run BEFORE the dpkg force-removes of essential packages
#      (ncurses-base, perl-base, login, passwd). Once `passwd` is gone the
#      `_apt` user disappears and apt can no longer read its keyring files —
#      llvm.sh's repo would then come back as unsigned. Verified locally.
#
#   3. No `apt-get autoremove` anywhere. Earlier attempts to use
#      `apt-mark manual` to protect libomp-22-dev failed: apt's autoremove
#      --purge ignores the manual flag once the dpkg force-removes have
#      broken the dep graph. We just don't autoremove. Image gains ~100MB of
#      leftover LLVM tooling (llvm-22-tools etc.) — acceptable for working
#      JIT.
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
    # ---- CVE trim: purge python3.13 FIRST, before LLVM is installed ----
    apt-get upgrade -y && \
    (apt-get purge -y python3.13 python3.13-minimal libpython3.13-minimal libpython3.13-stdlib || true) && \
    # ---- LLVM toolchain: install after the python cascade ----
    apt-get install -y --no-install-recommends wget gnupg lsb-release && \
    wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 22 && \
    apt-get install -y --no-install-recommends clang-22 libomp5 libomp-22-dev && \
    ln -sf /usr/bin/clang-22 /usr/bin/cc && \
    apt-get purge -y wget gnupg lsb-release && \
    # ---- Essential dpkg force-removes: LAST (these break apt afterwards) ----
    (dpkg --force-remove-essential --purge ncurses-base ncurses-bin || true) && \
    (dpkg --force-remove-essential --purge perl-base || true) && \
    (dpkg --force-remove-protected --force-remove-essential --purge login || true) && \
    (dpkg --purge passwd || true) && \
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
