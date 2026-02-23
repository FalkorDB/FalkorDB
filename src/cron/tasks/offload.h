/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// opaque
typedef struct OffloadTaskCtx OffloadTaskCtx ;

// the offload recurring task is responsible for scanning through each
// graph in the keyspace and offload attribute-sets to disk
//
// the task will only perform its work if the ratio between used memory and 
// available memory is lower than some specified threshold
// otherwise the task will be rescheduled to a later point in time
//
// attribute-sets are offload from the currently processed graph G while the
// task has acquired both G's write lock and the server's GIL

// create a new offload entities context
void *OffloadEntities_new (void) ;

// cron task entry point
// offloads all graphs in the keyspace one by one to disk
// the task is bounded by time (DEADLINE) after which it will exit
// and be rescheduled for future runs
//
// the task will quickly return if memory consumption is below a specified limit
// e.g. memory consumption < 65% quickly return otherwise start offloading
// returns true to increase task frequency, false to slowdown
bool OffloadEntities
(
	void *pdata  // task context
);

// free offload entities context
void OffloadEntities_free
(
	OffloadTaskCtx **ctx
);

