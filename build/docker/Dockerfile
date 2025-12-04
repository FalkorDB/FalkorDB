ARG BASE_IMAGE=falkordb/falkordb-compiler

ARG TARGETPLATFORM=linux/amd64

ARG BROWSER_TAG=latest

FROM $BASE_IMAGE AS compiler

FROM falkordb/falkordb-browser:$BROWSER_TAG AS browser

FROM redis:8.2.2

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

RUN apt-get update && apt-get install -y libgomp1 curl supervisor && \
    curl -sL https://deb.nodesource.com/setup_22.x -o nodesource_setup.sh && \
    bash nodesource_setup.sh && \
    mkdir -p /etc/supervisor/conf.d /var/log/supervisor && \
    apt-get install -y nodejs && \
    apt-get upgrade -y && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${FALKORDB_HOME}

COPY ./build/docker/run.sh ${FALKORDB_BIN_PATH}/run.sh
COPY ./build/docker/entrypoint.sh ${FALKORDB_BIN_PATH}/entrypoint.sh
COPY ./build/docker/gen-certs.sh ${FALKORDB_BIN_PATH}/gen-certs.sh
COPY ./build/docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY --from=compiler /FalkorDB/bin/linux*/src/falkordb.so ${FALKORDB_BIN_PATH}/falkordb.so

COPY --from=browser /app ${FALKORDB_BROWSER_PATH}

#Create necessary users and change ownership
RUN useradd -ms /bin/bash falkordb -U -u 1000 && \
    chown -R falkordb:falkordb ${FALKORDB_HOME} && \
    chmod -R 755 ${FALKORDB_HOME} && \
    useradd -ms /bin/bash nextjs -U -u 1001 && \
    chown -R nextjs:nextjs ${FALKORDB_BROWSER_PATH} && \
    chmod -R 755 ${FALKORDB_BROWSER_PATH}

ENV ARCH=${TARGETPLATFORM}

ENV TLS=0

ENV BROWSER=1
ENV REST_PORT=8080
ENV MCP_PORT=3001
ENV PORT=3000
ENV CYPHER=1

EXPOSE 6379/tcp 3001/tcp 8080/tcp 3000/tcp

ENV FALKORDB_ARGS="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000"

USER root

ENTRYPOINT ["/var/lib/falkordb/bin/entrypoint.sh"]

CMD [ "redis-server", "--loadmodule", "/var/lib/falkordb/bin/falkordb.so" ]
