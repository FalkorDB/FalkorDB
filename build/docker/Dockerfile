ARG BASE_IMAGE=falkordb/falkordb-compiler

ARG TARGETPLATFORM=linux/amd64

ARG BROWSER_TAG=latest

FROM $BASE_IMAGE AS compiler

FROM falkordb/falkordb-browser:$BROWSER_TAG AS browser


FROM redis:7.2.4

RUN apt-get update && apt-get install -y libgomp1 curl && \
    curl -sL https://deb.nodesource.com/setup_22.x -o nodesource_setup.sh && \
    bash nodesource_setup.sh && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /FalkorDB

COPY --from=compiler /FalkorDB/build/docker/run.sh /FalkorDB/build/docker/run.sh
COPY --from=compiler /FalkorDB/build/docker/gen-certs.sh /FalkorDB/build/docker/gen-certs.sh

COPY --from=compiler /FalkorDB/bin/linux*/src/falkordb.so /FalkorDB/bin/src/falkordb.so

COPY --from=browser /app /FalkorDBBrowser

ENV ARCH=${TARGETPLATFORM}

ENV TLS=0

ENV BROWSER=1

EXPOSE 6379/tcp

ENV FALKORDB_ARGS="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000"

CMD /FalkorDB/build/docker/run.sh

