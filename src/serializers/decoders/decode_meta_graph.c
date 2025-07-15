/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decode_graph.h"

// load a graph virtual key from RDB
void *RdbLoadMetaGraph
(
	RedisModuleIO *rdb,
	int encvar
) {
	// get key name from RDB
	const RedisModuleString *rm_key_name = RedisModule_GetKeyNameFromIO(rdb);
	const char *key_name = RedisModule_StringPtrLen(rm_key_name, NULL);

	// initialize SerializerIO from RDB
	SerializerIO io;
	if(encvar < 17) {
		io = SerializerIO_FromRedisModuleIO(rdb, false);
	} else {
		io = SerializerIO_FromBufferedRedisModuleIO(rdb, false);
	}

	// read encoded graph name
	char *graph_name = SerializerIO_ReadBuffer(io, NULL);
	ASSERT(graph_name != NULL);

	// graph name and meta graph key name should not match
	ASSERT(strcmp(key_name, graph_name) != 0);

	// load a graph from SerializerIO
	GraphContext *gc = SerializerLoadGraph(io, graph_name, encvar);

	// register meta key name for future deletion
	GraphDecodeContext_AddMetaKey(gc->decoding_context, key_name);

	// clean up
	rm_free(graph_name);
	SerializerIO_Free(&io);

	return gc;
}

