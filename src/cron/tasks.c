/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "cron.h"
#include "util/rmalloc.h"
#include "configuration/config.h"
#include "tasks/stream_finished_queries.h"

typedef struct RecurringTaskCtx {
	uint32_t when;
	uint32_t min_interval;
	uint32_t max_interval;
	Config_Option_Field field;
	bool (*task)(void*);
	void *(*new)(void*);
	void (*free)(void*);
	void *ctx;
} RecurringTaskCtx;

void CronTask_RecurringTaskFree
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
			CronTask_RecurringTaskFree, (void*)re_ctx) ;
}

void CronTask_AddStreamFinishedQueries() {
	//--------------------------------------------------------------------------
	// add query logging task
	//--------------------------------------------------------------------------

	// make sure info tracking is enabled
	bool info_enabled = false;
	if(Config_Option_get(Config_CMD_INFO, &info_enabled) && info_enabled) {
		RecurringTaskCtx *re_ctx = rm_malloc(sizeof(RecurringTaskCtx));
		re_ctx->new          = CronTask_newStreamFinishedQueries;
		re_ctx->task         = CronTask_streamFinishedQueries;
		re_ctx->free		 = rm_free;
		re_ctx->when         = 10;   // 10ms from now
		re_ctx->min_interval = 250;  // 250ms
		re_ctx->max_interval = 3000; // 3s

		// create task context
		StreamFinishedQueryCtx *ctx = rm_malloc(sizeof(StreamFinishedQueryCtx));
		ctx->graph_idx = 0;
		
		re_ctx->ctx = ctx;

		// add recurring task
		Cron_AddTask(0, CronTask_RecurringTask, CronTask_RecurringTaskFree, (void*)re_ctx);
	}
}

// add recurring tasks
void Cron_AddRecurringTasks(void) {
	CronTask_AddStreamFinishedQueries();
}

