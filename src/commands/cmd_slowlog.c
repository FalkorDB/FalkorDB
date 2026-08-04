/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../redismodule.h"
#include "../enterprise_api.h"
#include "../slow_log/slow_log.h"
#include "../graph/graphcontext.h"

// offloaded-graph-stub type, exported by the enterprise module via its
// shared API - resolved lazily; stays NULL when that module isn't loaded
// (community edition)
static GraphStubType_Get_t GraphStubType_Get = NULL ;

// usage:
// GRAPH.SLOWLOG G
// GRAPH.SLOWLOG G RESET
int Graph_Slowlog
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	//--------------------------------------------------------------------------
	// validations
	//--------------------------------------------------------------------------

	ASSERT(ctx  != NULL);
	ASSERT(argv != NULL);
	if(argc < 2 || argc > 3) {
		RedisModule_WrongArity(ctx);
		return REDISMODULE_OK;
	}

	// validate the subcommand before touching the keyspace at all - this is
	// a syntax check, applies the same way regardless of the key's type or
	// existence
	bool reset = false ;
	if (argc == 3) {
		const char *sub_cmd = RedisModule_StringPtrLen (argv[2], NULL) ;
		if (strcasecmp (sub_cmd, "reset") != 0) {
			RedisModule_ReplyWithError (ctx, "Unknown subcommand") ;
			return REDISMODULE_OK ;
		}
		reset = true ;
	}

	// get a hold of the graph key
	RedisModuleString *key = argv [1] ;

	// an offloaded stub has no slowlog history - answer directly without
	// paying for a disk load
	RedisModuleKey  *rkey = RedisModule_OpenKey (ctx, key, REDISMODULE_READ) ;
	RedisModuleType *type = RedisModule_ModuleTypeGetType (rkey) ;
	if (GraphStubType_Get == NULL) {
		GraphStubType_Get = RedisModule_GetSharedAPI (ctx, "GraphStubType_Get") ;
	}
	bool is_stub = (GraphStubType_Get != NULL && type == GraphStubType_Get ()) ;
	RedisModule_CloseKey (rkey) ;

	if (is_stub) {
		if (reset) {
			RedisModule_ReplyWithSimpleString (ctx, "OK") ;
		} else {
			RedisModule_ReplyWithArray (ctx, 0) ;
		}
		return REDISMODULE_OK ;
	}

	GraphContext *gc = NULL ;
	GraphContext_Retrieve (ctx, key, false, false, false, &gc) ;
	if (gc == NULL) {
		// if GraphContext is null, key access failed and an error been emitted
		return REDISMODULE_OK ;
	}

	SlowLog *slowlog = GraphContext_GetSlowLog (gc) ;

	if (reset) {
		SlowLog_Clear (slowlog) ;
		RedisModule_ReplyWithSimpleString (ctx, "OK") ;
	} else {
		// reply with slowlog
		SlowLog_Replay (slowlog, ctx) ;
	}

	GraphContext_DecreaseRefCount (gc) ;

	return REDISMODULE_OK ;
}

