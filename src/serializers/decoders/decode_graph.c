/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decoders.h"
#include "decode_graph.h"

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

// load a graph from SerializerIO
GraphContext *SerializerLoadGraph
(
	SerializerIO io,       // serializer
	const char *key_name,  // load graph under this key name
	int encvar             // encoder version
) {
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

	switch(encvar) {
		case 10:
			RdbLoadGraphContext_v10(io, gc);
			break;

		case 11:
			RdbLoadGraphContext_v11(io, gc);
			break;

		case 12:
			RdbLoadGraphContext_v12(io, gc);
			break;

		case 13:
			RdbLoadGraphContext_v13(io, gc);
			break;

		case 14: {
			RdbLoadGraphContext_v14(io, gc);
			break;
		}

		case 15: {
			RdbLoadGraphContext_v15(io, gc);
			break;
		}

		case 16: {
			RdbLoadGraphContext_v16(io, gc);
			break;
		}

		case 17: {
			RdbLoadGraphContext_v17(io, gc);
			break;
		}

		default:
			ASSERT(false && "attempted to read unsupported RedisGraph version from RDB file.");
			break;
	}

	return gc;
}

// load a graph from RDB
GraphContext *RdbLoadGraph
(
	RedisModuleIO *rdb,  // RDB
	int encvar           // encoder version
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
	rm_free(graph_name);

	// decode under key_name
	// note that key name might differ from graph name, this can happen in the
	// event of DUMP & RESTORE when the graph is restored under a different key
	GraphContext *gc = SerializerLoadGraph(io, key_name, encvar);

	SerializerIO_Free(&io);

	return gc;
}

