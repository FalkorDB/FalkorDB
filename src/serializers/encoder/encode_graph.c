/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "encode_graph.h"
#include "v15/encode_v15.h"
#include "../serializer_io.h"

void RdbSaveGraph(RedisModuleIO *rdb, void *value) {
	SerializerIO io = SerializerIO_FromRedisModuleIO(rdb);
	RdbSaveGraph_latest(io, value);
	SerializerIO_Free(&io);
}

