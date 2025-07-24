ARG BASE_IMAGE=falkordb/falkordb-compiler

ARG TARGETPLATFORM=linux/amd64

ARG BROWSER_TAG=latest

FROM $BASE_IMAGE AS compiler

FROM falkordb/falkordb-browser:$BROWSER_TAG AS browser


FROM redis:8.0.2

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

RUN apt-get update && apt-get install -y libgomp1 curl && \
    curl -sL https://deb.nodesource.com/setup_22.x -o nodesource_setup.sh && \
    bash nodesource_setup.sh && \
    apt-get install -y nodejs && \
    apt-get upgrade -y && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${FALKORDB_HOME}

COPY ./build/docker/run.sh ${FALKORDB_BIN_PATH}/run.sh
COPY ./build/docker/gen-certs.sh ${FALKORDB_BIN_PATH}/gen-certs.sh

COPY --from=compiler /FalkorDB/bin/linux*/src/falkordb.so ${FALKORDB_BIN_PATH}/falkordb.so

COPY --from=browser /app ${FALKORDB_BROWSER_PATH}

ENV ARCH=${TARGETPLATFORM}

ENV TLS=0

ENV BROWSER=1

EXPOSE 6379/tcp

ENV FALKORDB_ARGS="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000"

ENTRYPOINT ["/var/lib/falkordb/bin/run.sh"]

CMD [ "redis-server", "--loadmodule", "/var/lib/falkordb/bin/falkordb.so" ]
