/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decode_graph.h"
#include "current/v17/decode_v17.h"

// load a graph virtual key from RDB
GraphContext *RdbLoadMetaGraph
(
	RedisModuleIO *rdb
) {
	const RedisModuleString *rm_key_name = RedisModule_GetKeyNameFromIO(rdb);
	const char *key_name = RedisModule_StringPtrLen(rm_key_name, NULL);

	// initialize SerializerIO from RDB
	SerializerIO io = SerializerIO_FromBufferedRedisModuleIO(rdb, false);

	// read encoded graph name
	char *graph_name = SerializerIO_ReadBuffer(io, NULL);
	ASSERT(graph_name != NULL);

	// graph name and meta graph key name should not match
	ASSERT(strcmp(key_name, graph_name) != 0);

	// get graph object using the graph name
	GraphContext *gc = GraphContext_UnsafeGetGraphContext(graph_name);
	ASSERT(gc != NULL);
	
	// populate graph from RDB
	RdbLoadGraphContext_latest(io, gc);

	// register meta key name for future deletion
	GraphDecodeContext_AddMetaKey(gc->decoding_context, key_name);

	// clean up
	SerializerIO_Free(&io);

	return gc;
}

