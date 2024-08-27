export MODULE_DIR=/FalkorDB/bin/src

if [ ${BROWSER:-1} -eq 1 ]
then
    if [ -d /FalkorDBBrowser ]
    then
        cd /FalkorDBBrowser && HOSTNAME="0.0.0.0" node server.js &
    fi
fi

# Create /data directory if it does not exist
if [ ! -d /data ]
then
    mkdir /data
fi

if [ ${TLS:-0} -eq 1 ]
then
    /FalkorDB/build/docker/gen-certs.sh
    redis-server ${REDIS_ARGS} --protected-mode no \
                 --tls-port 6379 --port 0 \
                 --tls-cert-file ./tls/redis.crt \
                 --tls-key-file ./tls/redis.key \
                 --tls-ca-cert-file ./tls/ca.crt \
                 --tls-auth-clients no \
                 --dir /data \
                 --loadmodule ${MODULE_DIR}/falkordb.so ${FALKORDB_ARGS}
else
    redis-server ${REDIS_ARGS} --protected-mode no \
                 --dir /data \
                 --loadmodule ${MODULE_DIR}/falkordb.so ${FALKORDB_ARGS}
fi