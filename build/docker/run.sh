export MODULE_DIR=/FalkorDB/bin/src

if [[ ${!BROWSER} -eq 1 ]]
then
    if [ -d /FalkorDBBrowser ]
    then
        cd /FalkorDBBrowser && node server.js &
    fi
fi

if [[ ${!TLS} -eq 1 ]]
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