/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// forward declaration: the struct body is in recurring_task.c
typedef struct RecurringTaskCtx RecurringTaskCtx ;

// recurring task constructor
RecurringTaskCtx *RecurringTask_New
(
	uint32_t interval,       // reschedule every interval ms
	uint32_t min_interval,   // minimum rescheduling interval
	uint32_t max_interval,   // maximum rescheduling interval

	//--------------------------------------------------------------------------
	// hosted task callbacks
	//--------------------------------------------------------------------------

	bool (*task_cb)(void*),     // hosted task function pointer
	void (*destructor)(void*),  // hosted task's context destructor
	void *ctx                   // hosted task's context
);

// recurring task run function
// invoke the hosted task run function and determine if we need to speed up
// or slow down its next invocation
// the task re-schedules itself via Cron_AddTask upon each invocation
void RecurringTask_Run
(
	void *pdata  // recurring task context
);

// free recurring task and its hosted task context
void CronTask_RecurringTask_Free
(
	void *pdata  // recurring task context
);

