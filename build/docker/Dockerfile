FROM --platform=$TARGETPLATFORM falkordb/falkordb-build:latest as builder

FROM redis:7.2.4

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

ARG TARGETPLATFORM

WORKDIR /FalkorDB

COPY --from=builder /FalkorDB/build/docker/run.sh /FalkorDB/build/docker/run.sh
COPY --from=builder /FalkorDB/build/docker/gen-certs.sh /FalkorDB/build/docker/gen-certs.sh

COPY --from=builder /FalkorDB/bin/linux*/src/falkordb.so /FalkorDB/bin/src/falkordb.so

ENV ARCH=${TARGETPLATFORM}

EXPOSE 6379/tcp

ENV FALKORDB_ARGS="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000"

CMD /FalkorDB/build/docker/run.sh
                 
