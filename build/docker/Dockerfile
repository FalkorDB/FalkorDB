FROM swilly22/falkordb-build:latest

WORKDIR /FalkorDB

ADD . /FalkorDB

EXPOSE 6379/tcp

ENV MODULE_DIR="/FalkorDB/bin/linux-x64-release/src"

ENV FALKORDB_ARGS="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000"

RUN echo ${MODULE_DIR}

CMD redis-server --protected-mode no ${REDIS_ARGS} \
                 --loadmodule ${MODULE_DIR}/falkordb.so ${FALKORDB_ARGS}
                 
