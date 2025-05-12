/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "config.h"

int reconf_query_mem_cap_apply
(
	RedisModuleCtx *ctx,
	void *privdata,
	RedisModuleString **err
);

int reconf_max_queued_queries_apply
(
	RedisModuleCtx *ctx,
	void *privdata,
	RedisModuleString **err
);

int reconf_cmd_info_apply
(
	RedisModuleCtx *ctx,
	void *privdata,
	RedisModuleString **err
);

int reconf_deduplicate_strings_apply
(
	RedisModuleCtx *ctx,
	void *privdata,
	RedisModuleString **err
);

