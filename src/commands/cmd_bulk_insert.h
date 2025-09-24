/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../redismodule.h"
#include "../util/thpool/thpool.h"

/* Multi threaded bulk insert context. */
typedef struct {
	RedisModuleBlockedClient *bc;   // Blocked client.
	RedisModuleString **argv;
	int argc;
} BulkInsertContext;

BulkInsertContext *BulkInsertContext_New
(
	RedisModuleCtx *ctx,
	RedisModuleBlockedClient *bc,
	RedisModuleString **argv,
	int argc
);

void BulkInsertContext_Free
(
	BulkInsertContext *ctx
);

int Graph_BulkInsert(RedisModuleCtx *ctx, RedisModuleString **argv, int argc);

