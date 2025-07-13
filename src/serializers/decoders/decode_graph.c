/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decode_graph.h"
#include "current/v17/decode_v17.h"

static GraphContext *_GetOrCreateGraphContext
(
	const char *graph_name
) {
	GraphContext *gc = GraphContext_UnsafeGetGraphContext(graph_name);
	if(gc == NULL) {
		// new graph is being decoded
		// inform the module and create new graph context
		gc = GraphContext_New(graph_name);
		// while loading the graph
		// minimize matrix realloc and synchronization calls
		Graph_SetMatrixPolicy(gc->g, SYNC_POLICY_RESIZE);
	}

	return gc;
}

// load a graph from RDB
GraphContext *RdbLoadGraph
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

	// decode under key_name
	// it is possible for the graph_name not to match key_name
	// this can happen when restoring a graph under a different name
	// using the RESTORE command
	// e.g.
	// DUMP graph_a
	// RESTORE graph_b ...

	// get graph object using the key_name
	GraphContext *gc = _GetOrCreateGraphContext(key_name);
	ASSERT(gc != NULL);

	// populate graph from RDB
	RdbLoadGraphContext_latest(io, gc);

	// clean up
	rm_free(graph_name);
	SerializerIO_Free(&io);

	return gc;
}

