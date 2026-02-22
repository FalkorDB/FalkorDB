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
	RecurringTaskCtx *re_ctx = RecurringTask_New (
			10,    // every 10ms
			250,   // min interval 250ms
			3000,  // max interval 3s
			CronTask_streamFinishedQueries,
			rm_free,
			ctx) ;

	// add recurring task
	Cron_AddTask (0, RecurringTask_Run, CronTask_RecurringTask_Free,
			(void*)re_ctx) ;
}

// add data offloading task as a recurring cron task
static void CronTask_AddDataOffloadingTask(void) {
	// create task context
	void *ctx = CronTask_newOffloadEntities (NULL) ;

	// create a reuccring task
	RecurringTaskCtx *re_ctx = RecurringTask_New (
			10,    // every 10ms
			250,   // min interval 250ms
			3000,  // max interval 3s
			CronTask_offloadEntities,
			rm_free,
			ctx) ;

	// add recurring task
	Cron_AddTask (0, RecurringTask_Run, CronTask_RecurringTask_Free,
			(void*)re_ctx) ;
}

// add recurring tasks
void Cron_AddRecurringTasks(void) {
	CronTask_AddDataOffloadingTask () ;
	CronTask_AddStreamFinishedQueries () ;
}

