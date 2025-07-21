/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "../globals.h"
#include "../redismodule.h"
#include "../graph/graphcontext.h"

int Graph_List
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	ASSERT(ctx != NULL);

	if(argc != 1) {
		return RedisModule_WrongArity(ctx);
	}

	KeySpaceGraphIterator it;
	Globals_ScanGraphs(&it);
	RedisModule_ReplyWithArray(ctx, REDISMODULE_POSTPONED_LEN);

	// reply with each graph name
	uint64_t     n   = 0;
	GraphContext *gc = NULL;

	while((gc = GraphIterator_Next(&it)) != NULL) {
		const char *name = GraphContext_GetName(gc);
		RedisModule_ReplyWithStringBuffer(ctx, name, strlen(name));
		n++;
		GraphContext_DecreaseRefCount(gc);
	}

	RedisModule_ReplySetArrayLength(ctx, n);

	return REDISMODULE_OK;
}

