FROM --platform=$BUILDPLATFORM falkordb/falkordb-build:latest

ARG TARGETPLATFORM
ARG BUILDPLATFORM

WORKDIR /FalkorDB

ADD . /FalkorDB

EXPOSE 6379/tcp

ENV MODULE_DIR="/FalkorDB/bin/linux-${TARGETPLATFORM}-release/src"

ENV FALKORDB_ARGS="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000"

RUN echo ${MODULE_DIR}

CMD /FalkorDB/build/docker/run.sh
                 
