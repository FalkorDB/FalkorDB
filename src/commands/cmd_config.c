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

	// determine action
	int step;
	const char *action = RedisModule_StringPtrLen(argv[1], NULL);
	if(strcasecmp(action, "get") == 0) {
		step = 1;
	} else if(strcasecmp(action, "set") == 0) {
		step = 2;
	} else {
		RedisModule_ReplyWithErrorFormat(ctx, "ERR unknown subcommand '%s'.",
				action);
		return REDISMODULE_OK;
	}

	// add "GRAPH." prefix to each config key
	// e.g.
	// RESULTSET_MAX_SIZE > GRAPH.RESULTSET_MAX_SIZE

	for(int i = 2; i < argc; i+=step) {
		RedisModuleString *s = RedisModule_CreateStringPrintf(ctx, "%s%s",
				"graph.", RedisModule_StringPtrLen(argv[i], NULL));

		RedisModule_FreeString(ctx, argv[i]);
		argv[i] = s;
	}

	RedisModuleCallReply *reply = RedisModule_Call(ctx, "CONFIG", "v", argv+1, argc-1);

	RedisModule_ReplyWithCallReply(ctx, reply);

	RedisModule_FreeCallReply(reply);

	return REDISMODULE_OK;
}

