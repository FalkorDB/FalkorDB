/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include <string.h>
#include "RG.h"
#include "configuration/config.h"

int Graph_Config
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	// GRAPH.CONFIG <GET|SET> <NAME> [value]
	if(argc < 3) {
		return RedisModule_WrongArity(ctx);
	}

	// TODO: config get * should get only graph.* configs
	RedisModuleCallReply *reply = RedisModule_Call(ctx, "CONFIG", "v", argv+1, argc-1);

	RedisModule_ReplyWithCallReply(ctx, reply);

	RedisModule_FreeCallReply(reply);

	return REDISMODULE_OK;
}

