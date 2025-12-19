/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decode_graph.h"
#include "current/v19/decode_v19.h"

GraphContext *RdbLoadGraph
(
	RedisModuleIO *rdb
) {
	const RedisModuleString *rm_key_name = RedisModule_GetKeyNameFromIO(rdb);

	SerializerIO io = SerializerIO_FromBufferedRedisModuleIO(rdb, false);
	GraphContext *gc = RdbLoadGraphContext_latest(io, rm_key_name);
	SerializerIO_Free(&io);

	return gc;
}

