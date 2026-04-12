ARG BASE_IMAGE=falkordb/falkordb-compiler

ARG TARGETPLATFORM=linux/amd64

ARG BROWSER_TAG=latest

FROM $BASE_IMAGE AS compiler

FROM falkordb/falkordb-browser:$BROWSER_TAG AS browser


FROM node:24-bookworm-slim AS node

FROM redis:8.6.2

ENV FALKORDB_HOME=/var/lib/falkordb

ENV FALKORDB_DATA_PATH=${FALKORDB_HOME}/data

ENV FALKORDB_BIN_PATH=${FALKORDB_HOME}/bin

ENV FALKORDB_IMPORT_PATH=${FALKORDB_HOME}/import

ENV FALKORDB_TLS_PATH=${FALKORDB_HOME}/tls

ENV FALKORDB_BROWSER_PATH=${FALKORDB_HOME}/browser

# For backward compatibility, link previous bin path to new bin path
RUN mkdir -p /FalkorDB/build/docker && \
    mkdir -p /FalkorDB/bin/src && \
    mkdir -p /data

RUN ln -s ${FALKORDB_BIN_PATH} /FalkorDB/build/docker && \
    ln -s ${FALKORDB_BIN_PATH} /FalkorDB/bin/src && \
    ln -s ${FALKORDB_DATA_PATH} /data

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 ca-certificates && \
    apt-get upgrade -y && \
    # Remove packages not needed at runtime to reduce CVE surface
    (apt-get purge -y python3.13 python3.13-minimal libpython3.13-minimal libpython3.13-stdlib || true) && \
    apt-get autoremove -y && \
    # Remove ncurses (CVE-2025-69720 HIGH) and bash - not needed by redis-server or falkordb.so
    # run.sh uses #!/bin/sh (dash), so bash is not required
    (dpkg --force-remove-essential --purge ncurses-base ncurses-bin bash || true) && \
    (dpkg --force-remove-essential --force-depends --purge libtinfo6 || true) && \
    # Remove additional packages not needed at runtime
    (dpkg --force-remove-essential --purge perl-base || true) && \
    (dpkg --force-remove-protected --force-remove-essential --purge login || true) && \
    (dpkg --purge passwd || true) && \
    apt-get autoremove -y --purge 2>/dev/null || true && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy only Node.js runtime from official image (no npm, no curl/gnupg needed)
COPY --from=node /usr/local/bin/node /usr/local/bin/


WORKDIR ${FALKORDB_HOME}

COPY ./build/docker/run.sh ${FALKORDB_BIN_PATH}/run.sh
COPY ./build/docker/gen-certs.sh ${FALKORDB_BIN_PATH}/gen-certs.sh

COPY --from=compiler /FalkorDB/bin/linux*/falkordb.so ${FALKORDB_BIN_PATH}/falkordb.so

COPY --from=browser /app ${FALKORDB_BROWSER_PATH}

ENV ARCH=${TARGETPLATFORM}

ENV TLS=0

ENV BROWSER=1

EXPOSE 6379/tcp

ENV FALKORDB_ARGS="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000"

ENTRYPOINT ["/var/lib/falkordb/bin/run.sh"]

CMD [ "redis-server", "--loadmodule", "/var/lib/falkordb/bin/falkordb.so" ]
