/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../redismodule.h"
#include "../slow_log/slow_log.h"
#include "../graph/graphcontext.h"

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

	// get a hold of the graph key
	RedisModuleString *key = argv[1];
	GraphContext *gc = GraphContext_Retrieve(ctx, key, false, false);
	if(gc == NULL) {
		// if GraphContext is null, key access failed and an error been emitted
		return REDISMODULE_OK;
	}

	SlowLog *slowlog = GraphContext_GetSlowLog(gc);

	// handle subcommand e.g. GRAPH.SLOWLOG G RESET
	if(argc == 3) {
		const char *sub_cmd = RedisModule_StringPtrLen(argv[2], NULL);
		if(strcasecmp(sub_cmd, "reset") == 0) {
			SlowLog_Clear(slowlog);
			RedisModule_ReplyWithSimpleString(ctx, "OK");
		} else {
			// unknown subcommand
			RedisModule_ReplyWithError(ctx, "Unknown subcommand");
		}
		goto cleanup;
	}

	// reply with slowlog
	SlowLog_Replay(slowlog, ctx);

cleanup:
	GraphContext_DecreaseRefCount(gc);

	return REDISMODULE_OK;
}

