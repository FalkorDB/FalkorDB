/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../redismodule.h"
//#include "../graph/graphcontext.h"
//#include "../serializers/serializer_io.h"
//#include "../serializers/encoder/v17/encode_v17.h"
//#include "../serializers/decoders/current/v17/decode_v17.h"
//
//#include <stdio.h>
//#include <fcntl.h>
//#include <stdlib.h>
//#include <unistd.h>
//
//// clone a graph
//// this function executes on Redis main thread
////
//// usage:
//// GRAPH.DUMP <graph_id>
int Graph_Dump
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // command argument
	int argc                   // number of argument
) {
	return REDISMODULE_OK;
}
//	// validations
//	ASSERT(ctx  != NULL);
//	ASSERT(argv != NULL);
//
//	// expecting exactly 2 arguments:
//	// argv[0] command name
//	// argv[1] graph_id
//	if(argc != 2) {
//		return RedisModule_WrongArity(ctx);
//	}
//
//	//--------------------------------------------------------------------------
//	// validations
//	//--------------------------------------------------------------------------
//
//	// make sure src key is a graph
//	GraphContext *gc = GraphContext_Retrieve(ctx, argv[1], true, false);
//
//	// src key should be a graph
//	if(gc == NULL) {
//		// graph is missing, abort
//		RedisModule_ReplyWithError(ctx, "Failed to dump graph, graph doesn't exists");
//		return REDISMODULE_OK;
//	}
//
//	//--------------------------------------------------------------------------
//	// serialize graph to stream
//	//--------------------------------------------------------------------------
//
//	size_t size  = 0;
//	char *buffer = NULL;
//	SerializerIO io = NULL;
//
//    // open a memory stream
//    FILE *stream = open_memstream(&buffer, &size);
//	ASSERT(stream != NULL);
//
//	// create serializer
//	io = SerializerIO_FromStream(stream, true);
//	ASSERT(io != NULL);
//
//	RdbSaveGraph_latest(io, gc);
//
//	fflush(stream);  // must flush before using 'buffer'
//
//	// reply to caller
//	RedisModule_ReplyWithStringBuffer(ctx, buffer, size);
//
//cleanup:
//
//	// free serializer
//	if (io != NULL) {
//		SerializerIO_Free(&io);
//	}
//
//	// free buffer
//	if (buffer != NULL) {
//		free(buffer);
//	}
//
//	// decrease src graph ref-count
//	if (gc != NULL) {
//		GraphContext_DecreaseRefCount(gc);
//	}
//
//	return REDISMODULE_OK;
//}
//
