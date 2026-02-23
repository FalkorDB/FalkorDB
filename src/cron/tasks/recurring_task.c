/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../cron.h"
#include "../../util/rmalloc.h"

// recurring task context
typedef struct RecurringTaskCtx {
	uint32_t interval;      // task execution interval
	uint32_t min_interval;  // minimum rescheduling interval in ms
	uint32_t max_interval;  // maximum rescheduling interval in ms
	uint8_t  ref_count;     // task reference count

	//--------------------------------------------------------------------------
	// hosted task callbacks
	//--------------------------------------------------------------------------

	bool (*task_cb)(void *);      // task function pointer
	void (*destructor)(void **);  // task's context destructor
	void *ctx;                    // ctx passed to task and destructor callbacks
} RecurringTaskCtx;

// recurring task constructor
RecurringTaskCtx *RecurringTask_New
(
	uint32_t interval,       // reschedule every interval ms
	uint32_t min_interval,   // minimum rescheduling interval
	uint32_t max_interval,   // maximum rescheduling interval

	//--------------------------------------------------------------------------
	// hosted task callbacks
	//--------------------------------------------------------------------------

	bool (*task_cb)(void*),      // hosted task function pointer
	void (*destructor)(void**),  // hosted task's context destructor
	void *ctx                    // hosted task's context
) {
	ASSERT (task_cb      != NULL) ;
	ASSERT (destructor   != NULL) ;
	ASSERT (min_interval <= max_interval) ;

	RecurringTaskCtx *re_ctx = rm_malloc (sizeof (RecurringTaskCtx)) ;
	ASSERT (re_ctx != NULL) ;

	re_ctx->ctx          = ctx ;
	re_ctx->task_cb      = task_cb ;
	re_ctx->interval     = interval ;
	re_ctx->ref_count    = 1 ;
	re_ctx->destructor   = destructor ;
	re_ctx->min_interval = min_interval ;
	re_ctx->max_interval = max_interval ;

	return re_ctx ;
}

// free recurring task and its hosted task context
void CronTask_RecurringTask_Free
(
	void *pdata  // recurring task context
) {
	ASSERT (pdata != NULL) ;

	RecurringTaskCtx *ctx = (RecurringTaskCtx*)pdata ;
	ctx->ref_count-- ;

	// free only when reference count reach 0
	// call hosted task destructor
	if (ctx->ref_count  == 0    &&
		ctx->destructor != NULL &&
		ctx->ctx        != NULL) {
		ctx->destructor (&ctx->ctx) ;

		// free recurring task context
		rm_free (ctx) ;
	}
}

// recurring task run function
// invoke the hosted task run function and determine if we need to speed up
// or slow down its next invocation
// the task re-schedules itself via Cron_AddTask upon each invocation
void RecurringTask_Run
(
	void *pdata  // recurring task context
) {
	ASSERT (pdata != NULL) ;
	RecurringTaskCtx *re_ctx = (RecurringTaskCtx *)pdata ;

	// invoke hosted task
	// true: task has more work, reduce interval;
	// false: task is idle, increase interval
	bool speed_up = re_ctx->task_cb (re_ctx->ctx) ;

	// determine next invocation
	if (speed_up) {
		// reduce interval
		re_ctx->interval = (re_ctx->min_interval + re_ctx->interval) / 2 ;
	} else {
		// increase interval
		re_ctx->interval = (re_ctx->max_interval + re_ctx->interval) / 2 ;
	}

	//--------------------------------------------------------------------------
	// reschedule task
	//--------------------------------------------------------------------------

	// increase recurring task ref count, guard against free
	re_ctx->ref_count++ ;  

	// re-add task to CRON
	Cron_AddTask (re_ctx->interval, RecurringTask_Run,
			CronTask_RecurringTask_Free, re_ctx) ;
}

