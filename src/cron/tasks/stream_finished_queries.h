/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "util/simple_timer.h"

// task context
typedef struct {
	uint32_t graph_idx;        // last processed graph index
} StreamFinishedQueryCtx;

// create task context
void *CronTask_newStreamFinishedQueries
(
	void *pdata  // task context
);

// cron task
// stream finished queries for each graph in the keyspace
bool CronTask_streamFinishedQueries
(
	void *pdata  // task context
);

