FROM --platform=$BUILDPLATFORM falkordb/falkordb-build:latest

ARG TARGETPLATFORM
ARG BUILDPLATFORM

WORKDIR /FalkorDB

ADD . /FalkorDB

RUN make

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
      export MODULE_DIR=/FalkorDB/bin/linux-x64-release/src; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
      export MODULE_DIR=/FalkorDB/bin/linux-arm64v8-release/src; \
    else \
      echo "Platform not supported"; \
    fi && \
    echo "MY_VARIABLE=$MY_VARIABLE" >> /etc/environment

EXPOSE 6379/tcp

ENV MODULE_DIR="/FalkorDB/bin/linux-${TARGETPLATFORM}-release/src"

ENV FALKORDB_ARGS="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000"

RUN echo ${MODULE_DIR}

CMD /FalkorDB/build/docker/run.sh
                 
