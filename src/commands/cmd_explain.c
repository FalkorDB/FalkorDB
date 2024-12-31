/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "cmd_context.h"
#include "../globals.h"
#include "../query_ctx.h"
#include "execution_ctx.h"
#include "../index/index.h"
#include "../util/rmalloc.h"
#include "../errors/errors.h"
#include "../execution_plan/execution_plan.h"

// builds an execution plan but does not execute it
// reports plan back to the client
// Args:
// argv[1] graph name
// argv[2] query
void Graph_Explain(void *args) {
	bool lock_acquired = false;
	CommandCtx     *command_ctx = (CommandCtx *)args;
	RedisModuleCtx *ctx         = CommandCtx_GetRedisCtx(command_ctx);
	GraphContext   *gc          = CommandCtx_GetGraphContext(command_ctx);
	ExecutionCtx   *exec_ctx    = NULL;
	QueryCtx       *query_ctx   = QueryCtx_GetQueryCtx();

	QueryCtx_SetGlobalExecutionCtx(command_ctx);
	Globals_TrackCommandCtx(command_ctx);

	// retrieve the required execution items and information:
	// 1. Execution plan
	// 2. Whether these items were cached or not
	bool           cached = false;
	ExecutionPlan  *plan  = NULL;
	exec_ctx  =  ExecutionCtx_FromQuery(command_ctx->query);
	if (exec_ctx == NULL) {
		query_ctx->status = QueryExecutionStatus_FAILURE;
		goto cleanup;
	}

	plan = exec_ctx->plan;
	ExecutionType exec_type = exec_ctx->exec_type;

	if(exec_type == EXECUTION_TYPE_INDEX_CREATE) {
		RedisModule_ReplyWithSimpleString(ctx, "Create Index");
		goto cleanup;
	} else if(exec_type == EXECUTION_TYPE_INDEX_DROP) {
		RedisModule_ReplyWithSimpleString(ctx, "Drop Index");
		goto cleanup;
	}

	CRWGuard guard = Graph_AcquireReadLock(gc->g);
	lock_acquired = true;

	ExecutionPlan_PreparePlan(plan);
	ExecutionPlan_Init(plan);       // Initialize the plan's ops.

	if (ErrorCtx_EncounteredError()) {
		query_ctx->status = QueryExecutionStatus_FAILURE;
		goto cleanup;
	}

	ExecutionPlan_Print(plan, ctx); // Print the execution plan.

cleanup:
	if(ErrorCtx_EncounteredError()) ErrorCtx_EmitException();
	if(lock_acquired) Graph_ReleaseLock(gc->g, guard);
	ExecutionCtx_Free(exec_ctx);
	GraphContext_DecreaseRefCount(gc);
	Globals_UntrackCommandCtx(command_ctx);
	CommandCtx_UnblockClient(command_ctx);
	CommandCtx_Free(command_ctx);
	QueryCtx_Free(); // Reset the QueryCtx and free its allocations.
	ErrorCtx_Clear();
}

