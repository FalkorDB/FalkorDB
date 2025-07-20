/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decode_graph.h"

// extract graph id from meta key name
static char *ExtractGraphID
(
	const char *key
) {
	ASSERT(key != NULL);

	// a graph meta key name is in one of two possible forms:
	// 1. graphID_UUID
	// 2. {graphID}graphID_UUID
	//
	// where the UUID part is a 36 chars in length

	// check if key starts with '{'
	int start = 0;
	int end   = strlen(key) - 37;  // UUID + '_'

	ASSERT(end > start);

	if (key[0] == '{') {
		// search for closing '}'
		char *pos = strchr(key, '}');
		if (pos != NULL) {
			start = (pos - key) + 1;
		}
	}

	size_t len = end - start;
	char *graph_id = rm_malloc(len + 1);

	memcpy(graph_id, key + start, len);
	graph_id[len] = '\0';

	return graph_id;
}

// load a graph virtual key from RDB
void *RdbLoadMetaGraph
(
	RedisModuleIO *rdb,
	int encvar
) {
	// get key name from RDB
	const RedisModuleString *rm_key_name = RedisModule_GetKeyNameFromIO(rdb);
	const char *key_name = RedisModule_StringPtrLen(rm_key_name, NULL);

	char *graph_name = ExtractGraphID(key_name);

	// initialize SerializerIO from RDB
	SerializerIO io;
	if(encvar < 17) {
		io = SerializerIO_FromRedisModuleIO(rdb, false);
	} else {
		io = SerializerIO_FromBufferedRedisModuleIO(rdb, false);
	}

	// load a graph from SerializerIO
	GraphContext *gc = SerializerLoadGraph(io, graph_name, encvar);

	// register meta key name for future deletion
	GraphDecodeContext_AddMetaKey(gc->decoding_context, key_name);

	// clean up
	rm_free(graph_name);
	SerializerIO_Free(&io);

	return gc;
}

