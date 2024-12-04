/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "decode_graph.h"
#include "current/v16/decode_v16.h"

GraphContext *RdbLoadGraph(RedisModuleIO *rdb) {
	const RedisModuleString *rm_key_name = RedisModule_GetKeyNameFromIO(rdb);

	SerializerIO io = SerializerIO_FromRedisModuleIO(rdb);
	GraphContext *gc = RdbLoadGraphContext_latest(io, rm_key_name);
	SerializerIO_Free(&io);

	return gc;
}

