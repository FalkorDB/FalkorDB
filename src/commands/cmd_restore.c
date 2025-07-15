/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../graph/graphcontext.h"
#include "../serializers/serializer_io.h"

extern RedisModuleType *GraphContextRedisModuleType;

// restore a graph from binary representation
// this command is the counter part of GRAPH.COPY
// which replicates the cloned graph via GRAPH.RESTORE
//
// usage:
// GRAPH.RESTORE <graph> <payload>
//
// this function is ment to execute on redis main thread
int Graph_Restore
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // command argument
	int argc                   // number of argument
) {
	return REDISMODULE_OK;
}
	// validations
//	ASSERT(ctx  != NULL);
//	ASSERT(argv != NULL);
//
//	// expecting exactly 3 arguments:
//	// argv[0] command name
//	// argv[1] graph key
//	// argv[2] graph payload
//	if(argc != 3) {
//		return RedisModule_WrongArity(ctx);
//	}
//
//	// TODO: reject GRAPH.RESTORE if caller isn't the master
//
//	//--------------------------------------------------------------------------
//	// make sure graph key doesn't exists
//	//--------------------------------------------------------------------------
//
//	RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1], REDISMODULE_READ);
//	int key_type = RedisModule_KeyType(key);
//	RedisModule_CloseKey(key);
//
//	// key exists, fail
//	if(key_type != REDISMODULE_KEYTYPE_EMPTY) {
//		RedisModule_ReplyWithError(ctx, "restore graph failed, key already exists");
//		return REDISMODULE_OK;
//	}
//
//	//--------------------------------------------------------------------------
//	// decode payload
//	//--------------------------------------------------------------------------
//	
//	// create memory stream
//	size_t len;
//	const char *payload = RedisModule_StringPtrLen(argv[2], &len);
//
//	FILE *stream = fmemopen((void*)payload, len, "r");
//	ASSERT(stream != NULL);
//
//	SerializerIO io = SerializerIO_FromStream(stream, false);
//
//	// decode graph
//	GraphContext *gc = GraphContext_Retrieve(ctx, argv[1], false, true);
//	RdbLoadGraphContext_latest(io, gc);
//	ASSERT(gc != NULL);
//
//	// add graph to keyspace
//	key = RedisModule_OpenKey(ctx, argv[1], REDISMODULE_WRITE);
//
//	// set value in key
//	RedisModule_ModuleTypeSetValue(key, GraphContextRedisModuleType, gc);
//
//	RedisModule_CloseKey(key);
//
//	// register graph context for BGSave
//	GraphContext_RegisterWithModule(gc);
//
//	//--------------------------------------------------------------------------
//	// clean up
//	//--------------------------------------------------------------------------
//
//	SerializerIO_Free(&io);
//
//	fclose(stream);
//
//	RedisModule_ReplyWithCString(ctx, "OK");
//
//	return REDISMODULE_OK;
//}

