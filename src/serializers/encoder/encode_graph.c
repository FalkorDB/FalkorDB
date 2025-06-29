/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "encode_graph.h"
#include "v18/encode_v18.h"
#include "../serializer_io.h"

void RdbSaveGraph
(
	RedisModuleIO *rdb,
	void *value
) {
	SerializerIO io = SerializerIO_FromBufferedRedisModuleIO(rdb, true);
	RdbSaveGraph_latest(io, value);
	SerializerIO_Free(&io);
}

