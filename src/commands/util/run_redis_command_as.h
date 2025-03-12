/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "../../redismodule.h"

extern const char *ADMIN_USER;

typedef int (*RedisCommandAsUserFunc)(RedisModuleCtx *ctx,
	RedisModuleString **argv, int argc, const char *username, void *privdata);

int run_redis_command_as(RedisModuleCtx *ctx, RedisModuleString **argv,
	int argc, RedisCommandAsUserFunc cmd, const char *username, void *privdata);

