/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#define SLOW_LOG_SIZE 10

#include <pthread.h>

#include "../util/heap.h"
#include "../redismodule.h"
#include "../../deps/rax/rax.h"

// Slowlog item.
typedef struct {
    char *cmd;          // Redis command.
    time_t time;        // Item creation time.
	char *query;        // Query.
	double latency;     // How much time query was processed.
} SlowLogItem;

// Slowlog, maintains N slowest queries.
typedef struct {
     uint count;                // Length of lookup, min_heap and locks arrays.
     rax **lookup;              // Array of item lookup table.
     heap_t **min_heap;         // Array of minimum heap of items.
     pthread_mutex_t *locks;    // Array of locks.
} SlowLog;

// Create a new slowlog.
SlowLog *SlowLog_New();

// Introduce item to slow log.
void SlowLog_Add
(
	SlowLog *slowlog,			// slowlog to add entry to
	const char *cmd,			// command being logged
	const char *query,			// query being logged
	double latency,				// command latency
	time_t *time				// optional time command was issued
);

// clear all entries from slowlog
void SlowLog_Clear
(
	SlowLog *slowlog  // slowlog to clear
);

// Replies with slow log content.
void SlowLog_Replay
(
	const SlowLog *slowlog,
	RedisModuleCtx *ctx
);

// Free slowlog.
void SlowLog_Free
(
	SlowLog *slowlog
);

