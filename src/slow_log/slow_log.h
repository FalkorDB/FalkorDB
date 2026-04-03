/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../redismodule.h"

// Forward declaration (incomplete type)
typedef struct SlowLog SlowLog;

// create a new slowlog
SlowLog *SlowLog_New (void);

// introduce item to slow log
void SlowLog_Add
(
	SlowLog *slowlog,   // slowlog to add entry to
	const char *cmd,    // command
	const char *query,  // query being logged
	int params_len,     // params byte length
	double latency,     // command latency
	uint64_t time       // seconds since UNIX epoch
);

// clear all entries from slowlog
void SlowLog_Clear
(
	SlowLog *slowlog  // slowlog to clear
);

// reports slowlog content
void SlowLog_Replay
(
	SlowLog *slowlog,    // slowlog
	RedisModuleCtx *ctx  // redis module context
);

// free slowlog
void SlowLog_Free
(
	SlowLog *slowlog  // slowlog to free
);

