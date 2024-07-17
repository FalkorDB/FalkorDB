ARG BASE_IMAGE=falkordb/falkordb-compiler
ARG TARGETPLATFORM=linux/amd64
FROM --platform=${TARGETPLATFORM} $BASE_IMAGE AS compiler


FROM falkordb/falkordb-browser AS browser


FROM redis:7.2.4

RUN apt-get update && apt-get install -y libgomp1 nodejs && rm -rf /var/lib/apt/lists/*

WORKDIR /FalkorDB

COPY --from=compiler /FalkorDB/build/docker/run.sh /FalkorDB/build/docker/run.sh
COPY --from=compiler /FalkorDB/build/docker/gen-certs.sh /FalkorDB/build/docker/gen-certs.sh

COPY --from=compiler /FalkorDB/bin/linux*/src/falkordb.so /FalkorDB/bin/src/falkordb.so

COPY --from=browser /app /FalkorDBBrowser

# Args do not transfer between layers
ARG TARGETPLATFORM=linux/amd64
ENV ARCH=${TARGETPLATFORM}

ENV TLS=0

ENV BROWSER=1

EXPOSE 6379/tcp

ENV FALKORDB_ARGS="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000"

CMD /FalkorDB/build/docker/run.sh

