/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "cron/cron.h"
#include "util/rmalloc.h"
#include "reconf_handler.h"
#include "util/thpool/pools.h"

int reconf_query_mem_cap_apply
(
	RedisModuleCtx *ctx,
	void *privdata,
	RedisModuleString **err
) {
	int64_t query_mem_capacity;
	bool res = Config_Option_get(Config_QUERY_MEM_CAPACITY, &query_mem_capacity);
	ASSERT(res);
	rm_set_mem_capacity(query_mem_capacity);

	return REDISMODULE_OK;
}

int reconf_max_queued_queries_apply
(
	RedisModuleCtx *ctx,
	void *privdata,
	RedisModuleString **err
) {
	uint64_t max_queued_queries;
	bool res = Config_Option_get(Config_MAX_QUEUED_QUERIES, &max_queued_queries);
	ASSERT(res);
	ThreadPools_SetMaxPendingWork(max_queued_queries);

	return REDISMODULE_OK;
}

int reconf_cmd_info_apply
(
	RedisModuleCtx *ctx,
	void *privdata,
	RedisModuleString **err
) {
	bool info_enabled;
	bool res = Config_Option_get(Config_CMD_INFO, &info_enabled);
	ASSERT(res);
	if(info_enabled) {
		CronTask_AddStreamFinishedQueries();
	}

	return REDISMODULE_OK;
}

int reconf_deduplicate_strings_apply
(
	RedisModuleCtx *ctx,
	void *privdata,
	RedisModuleString **err
) {
	bool enabled;
	bool res = Config_Option_get(Config_DEDUPLICATE_STRINGS, &enabled);
	ASSERT(res);

	extern bool USE_STRING_POOL;  // defined in src/value.c
	USE_STRING_POOL = enabled;

	return REDISMODULE_OK;
}

