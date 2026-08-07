/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "encode_graph.h"
#include "v20/encode_v20.h"
#include "../serializer_io.h"

void RdbSaveGraph
(
	RedisModuleIO *rdb,
	void *value
) {
	SerializerIO io = SerializerIOv2_FromBufferedRedisModuleIO(rdb, true);
	RdbSaveGraph_latest(io, value);
	SerializerIO_Free(&io);
}

