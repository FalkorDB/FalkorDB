if [ "$ARCH" = "linux/amd64" ]
then
    export MODULE_DIR=/FalkorDB/bin/linux-x64-release/src
elif [ "$ARCH" = "linux/arm64" ]
then
    export MODULE_DIR=/FalkorDB/bin/linux-arm64v8-release/src
else
    echo "Platform not supported"
fi

if [ ${TLS} -eq 1 ]
then
    /FalkorDB/build/docker/gen-certs.sh
    redis-server --protected-mode no ${REDIS_ARGS} \
                 --tls-port 6379 --port 0 \
                 --tls-cert-file ./tls/redis.crt \
                 --tls-key-file ./tls/redis.key \
                 --tls-ca-cert-file ./tls/ca.crt \
                 --tls-auth-clients no \
                 --loadmodule ${MODULE_DIR}/falkordb.so ${FALKORDB_ARGS}
else
    redis-server --protected-mode no ${REDIS_ARGS} \
                 --loadmodule ${MODULE_DIR}/falkordb.so ${FALKORDB_ARGS}
fi