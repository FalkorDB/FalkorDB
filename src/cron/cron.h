/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <sys/types.h>

// CRON is a task scheduler
// a task is defined by:
// when it should run; delta in ms from the time it's introduced
// a callback to call when it is time to execute the task
// and an optional private data passed to the callback

// task callback functions
typedef void (*CronTaskCB)(void *pdata);    // task work function
typedef void (*CronTaskFree)(void *pdata);  // task private data free function
typedef uintptr_t CronTaskHandle;

// start CRON, should be called once
bool Cron_Start(void);

// stop CRON
void Cron_Stop(void);

// add recurring tasks
void Cron_AddRecurringTasks(void);

// add stream finished queries task
void CronTask_AddStreamFinishedQueries();

// create a new CRON task
CronTaskHandle Cron_AddTask
(
	uint when,          // number of miliseconds until task invocation
	CronTaskCB work,    // callback to call when task is due
	CronTaskFree free,  // [optional] task private data free function
	void *pdata         // [optional] private data to pass to callback
);

// aborts the CRON task passed (if found).
// this function wait until the task is completed if it has
// already started at the moment of invocation.
bool Cron_AbortTask
(
	CronTaskHandle t
);

