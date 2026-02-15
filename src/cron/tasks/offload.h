/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// the offload recurring task is responsible for scanning through each
// graph in the keyspace and offload attribute-sets to disk
//
// the task will only perform its work if the ratio between used memory and 
// available memory is lower than some specified threshold
// otherwise the task will be rescheduled to a later point in time
//
// attribute-sets are offload from the currently processed graph G while the
// task has acquired both G's write lock and the server's GIL

void *CronTask_newOffloadEntities
(
	void *pdata  // task context
);

bool CronTask_offloadEntities
(
	void *pdata  // task context
);

