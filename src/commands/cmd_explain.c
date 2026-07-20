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
#include "../graph/graphcontext_retrieve.h"
#include "../execution_plan/execution_plan.h"

// builds an execution plan but does not execute it
// reports plan back to the client
// Args:
// argv[1] graph name
// argv[2] query
//
// optnone: see the identical comment on _query in cmd_query.c - the same
// "goto cleanup" / llvm.assume miscompilation risk applies here now that
// this function also has a GraphContext_RetrieveOrQueue failure path
__attribute__((optnone))
void Graph_Explain
(
	void *args
) {
	bool lock_acquired = false ;
	CommandCtx     *command_ctx = (CommandCtx *)args ;
	RedisModuleCtx *ctx         = CommandCtx_GetRedisCtx (command_ctx) ;
	GraphContext   *gc          = CommandCtx_GetGraphContext (command_ctx) ;
	ExecutionCtx   *exec_ctx    = NULL ;
	QueryCtx       *query_ctx   = QueryCtx_GetQueryCtx () ;

	Globals_TrackCommandCtx (command_ctx) ;

	if (gc == NULL) {
		GraphRetrieveStatus status = GraphContext_RetrieveOrQueue (ctx,
				command_ctx->rm_graph_name, true, false, command_ctx, &gc) ;

		if (status == GraphRetrieve_LOADING) {
			// parked behind another thread's in-flight load of this graph;
			// CommandCtx_ResumeAfterGraphLoad will resubmit (or fail) this
			// command once that load resolves - undo this attempt's
			// thread-local setup, leave the CommandCtx / blocked client alone
			Globals_UntrackCommandCtx (command_ctx) ;
			QueryCtx_Free () ;
			return ;
		}

		if (status != GraphRetrieve_RETRIEVED) {
			goto cleanup ;
		}

		CommandCtx_SetGraphContext (command_ctx, gc) ;
	}

	QueryCtx_SetGlobalExecutionCtx (command_ctx) ;

	// retrieve the required execution items and information:
	// 1. Execution plan
	// 2. Whether these items were cached or not
	bool           cached = false ;
	ExecutionPlan  *plan  = NULL ;

	exec_ctx = ExecutionCtx_FromQuery (command_ctx) ;
	if (exec_ctx == NULL) {
		query_ctx->status = QueryExecutionStatus_FAILURE ;
		goto cleanup ;
	}

	plan = exec_ctx->plan ;
	ExecutionType exec_type = exec_ctx->exec_type ;

	if (exec_type == EXECUTION_TYPE_INDEX_CREATE) {
		RedisModule_ReplyWithSimpleString (ctx, "Create Index") ;
		goto cleanup ;
	} else if (exec_type == EXECUTION_TYPE_INDEX_DROP) {
		RedisModule_ReplyWithSimpleString (ctx, "Drop Index") ;
		goto cleanup ;
	}

	GraphContext_AcquireReadLock (gc) ;
	lock_acquired = true ;

	ExecutionPlan_PreparePlan (plan) ;
	ExecutionPlan_Init (plan) ;       // initialize the plan's ops

	if (ErrorCtx_EncounteredError ()) {
		query_ctx->status = QueryExecutionStatus_FAILURE ;
		goto cleanup ;
	}

	ExecutionPlan_Print (plan, ctx) ; // print the execution plan

cleanup:

	if (ErrorCtx_EncounteredError ()) {
		ErrorCtx_EmitException () ;
	}

	if (lock_acquired) {
		GraphContext_ReleaseLock (gc) ;
	}

	ExecutionCtx_Free (exec_ctx) ;

	if (gc) {
		GraphContext_DecreaseRefCount (gc) ;
	}

	Globals_UntrackCommandCtx (command_ctx) ;
	CommandCtx_UnblockClient (command_ctx) ;
	CommandCtx_Free (command_ctx) ;
	QueryCtx_Free () ; // reset the QueryCtx and free its allocations
	ErrorCtx_Clear () ;
}

