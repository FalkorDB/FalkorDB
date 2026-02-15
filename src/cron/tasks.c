/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "cron.h"
#include "tasks/tasks.h"
#include "util/rmalloc.h"
#include "configuration/config.h"

typedef struct RecurringTaskCtx {
	uint32_t when;          // next reschedule in ms
	uint32_t min_interval;  // minimum rescheduling interval
	uint32_t max_interval;  // maximum rescheduling interval
	bool (*task)(void*);    // task function pointer
	void *(*new)(void*);    // task's context constructor
	void (*free)(void*);    // task's context destructor
	void *ctx;              // task's context
} RecurringTaskCtx;

RecurringTaskCtx *RecurringTaskCtx_New
(
	uint32_t when,           // next reschedule in ms
	uint32_t min_interval,   // minimum rescheduling interval
	uint32_t max_interval,   // maximum rescheduling interval
	bool (*task_fp)(void*),   // task function pointer
	void *(*new_fp)(void*),  // task's context constructor
	void (*free_fp)(void*),  // task's context destructor
	void *ctx                // task's context
) {
	ASSERT (new_fp  != NULL) ;
	ASSERT (free_fp != NULL) ;
	ASSERT (task_fp != NULL) ;
	ASSERT (min_interval <= max_interval) ;

	RecurringTaskCtx *re_ctx = rm_malloc (sizeof (RecurringTaskCtx)) ;

	re_ctx->ctx          = ctx;
	re_ctx->new          = new_fp ;
	re_ctx->when         = when ;
	re_ctx->task         = task_fp ;
	re_ctx->free         = free_fp ;
	re_ctx->min_interval = min_interval ;
	re_ctx->max_interval = max_interval ;

	return re_ctx ;
}

void CronTask_RecurringTask_Free
(
	void *pdata
) {
	ASSERT(pdata != NULL);
	RecurringTaskCtx *current_ctx = (RecurringTaskCtx*)pdata;
	current_ctx->free(current_ctx->ctx);
	rm_free(current_ctx);
}

void CronTask_RecurringTask
(
	void *pdata
) {
	ASSERT(pdata != NULL);
	RecurringTaskCtx *current_ctx = (RecurringTaskCtx*)pdata;
	bool speed_up = current_ctx->task(current_ctx->ctx);	

	RecurringTaskCtx *re_ctx = rm_malloc(sizeof(RecurringTaskCtx));
	*re_ctx = *current_ctx;
	re_ctx->ctx = re_ctx->new(re_ctx->ctx);

	// determine next invocation
	if(speed_up) {
		// reduce delay, lower limit: 250ms
		re_ctx->when = (re_ctx->min_interval + re_ctx->when) / 2;
	} else {
		// increase delay, upper limit: 3sec
		re_ctx->when = (re_ctx->max_interval + re_ctx->when) / 2;
	}

	// re-add task to CRON
	Cron_AddTask (re_ctx->when, CronTask_RecurringTask,
			CronTask_RecurringTask_Free, (void*)re_ctx) ;
}

void CronTask_AddStreamFinishedQueries(void) {
	//--------------------------------------------------------------------------
	// add query logging task
	//--------------------------------------------------------------------------

	// make sure info tracking is enabled
	bool info_enabled = false;
	if (!(Config_Option_get (Config_CMD_INFO, &info_enabled) && info_enabled)) {
		// info tracking disabled, quickly return
		return ;
	}

	// create task context
	StreamFinishedQueryCtx *ctx = rm_malloc (sizeof (StreamFinishedQueryCtx)) ;
	ctx->graph_idx = 0 ;

	// create a reuccring task
	RecurringTaskCtx *re_ctx = RecurringTaskCtx_New (
			10,    // 10ms from now
			250,   // min interval 250ms
			3000,  // max interval 3s
			CronTask_streamFinishedQueries,
			CronTask_newStreamFinishedQueries,
			rm_free,
			ctx) ;

	// add recurring task
	Cron_AddTask (0, CronTask_RecurringTask, CronTask_RecurringTask_Free,
			(void*)re_ctx) ;
}

// add data offloading task as a recurring cron task
static void CronTask_AddDataOffloadingTask(void) {
	// create task context
	void *ctx = CronTask_newOffloadEntities (NULL) ;

	// create a reuccring task
	RecurringTaskCtx *re_ctx = RecurringTaskCtx_New (
			10,    // 10ms from now
			250,   // min interval 250ms
			3000,  // max interval 3s
			CronTask_offloadEntities,
			CronTask_newOffloadEntities,
			rm_free,
			ctx) ;

	// add recurring task
	Cron_AddTask (0, CronTask_RecurringTask, CronTask_RecurringTask_Free,
			(void*)re_ctx) ;
}

// add recurring tasks
void Cron_AddRecurringTasks(void) {
	CronTask_AddDataOffloadingTask () ;
	CronTask_AddStreamFinishedQueries () ;
}

