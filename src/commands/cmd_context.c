/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "cmd_context.h"
#include "../globals.h"
#include "../util/rmalloc.h"
#include "../slow_log/slow_log.h"
#include "../util/blocked_client.h"

#include <stdatomic.h>

// create a new command context
CommandCtx *CommandCtx_New
(
	RedisModuleCtx *ctx,           // redis module context
	RedisModuleBlockedClient *bc,  // blocked client
	RedisModuleString *cmd_name,   // command to execute
	RedisModuleString *query,      // query string
	GraphContext *graph_ctx,       // graph context
	ExecutorThread thread,         // which thread executes this command
	bool replicated_command,       // whether this instance was spawned by a replication command
	bool compact,                  // whether this query was issued with the compact flag
	long long timeout,             // the query timeout, if specified
	bool timeout_rw,               // apply timeout on both read and write queries
	uint64_t received_ts,          // command received at this UNIX timestamp
	simple_timer_t timer,          // stopwatch started upon command received
	bolt_client_t *bolt_client     // BOLT client
) {
	ASSERT (query    != NULL) ;
	ASSERT (cmd_name != NULL) ;

	CommandCtx *context = rm_calloc (1, sizeof (CommandCtx)) ;

	context->bc                 = bc;
	context->ctx                = ctx;
	context->thread             = thread;
	context->compact            = compact;
	context->timeout            = timeout;
	context->ref_count          = ATOMIC_VAR_INIT(1);
	context->graph_ctx          = graph_ctx;
	context->timeout_rw         = timeout_rw;
	context->bolt_client        = bolt_client;
	context->received_ts        = received_ts;
	context->replicated_command = replicated_command;

	simple_timer_copy(timer, context->timer);

	// retain command name
	// threaded modules that reference retained strings from other threads
	// must explicitly trim the allocation as soon as the string is retained.
	// Not doing so may result with automatic trimming which is not thread safe.
	RedisModule_RetainString (ctx, cmd_name) ;
	RedisModule_TrimStringAllocation (cmd_name) ;

	context->rm_command_name = cmd_name ;
	context->command_name = RedisModule_StringPtrLen (cmd_name, NULL) ;

	// retain query
	RedisModule_RetainString (ctx, query) ;
	RedisModule_TrimStringAllocation (query) ;

	context->rm_query = query ;
	context->query = RedisModule_StringPtrLen (query, &context->query_len) ;

	return context;
}

// increment command context reference count
void CommandCtx_Incref
(
	CommandCtx *cmd_ctx
) {
	ASSERT(cmd_ctx != NULL);

	// atomicly increment reference count
	atomic_fetch_add(&cmd_ctx->ref_count, 1);
}

RedisModuleCtx *CommandCtx_GetRedisCtx
(
	CommandCtx *cmd_ctx
) {
	ASSERT(cmd_ctx != NULL);
	// either we already have a context or block client is set
	if(cmd_ctx->ctx) {
		return cmd_ctx->ctx;
	}

	ASSERT(cmd_ctx->bc != NULL);

	cmd_ctx->ctx = RedisModule_GetThreadSafeContext(cmd_ctx->bc);
	return cmd_ctx->ctx;
}

bolt_client_t *CommandCtx_GetBoltClient
(
	CommandCtx *cmd_ctx
) {
	ASSERT(cmd_ctx != NULL);
	return cmd_ctx->bolt_client;
}

RedisModuleBlockedClient *CommandCtx_GetBlockingClient
(
	const CommandCtx *cmd_ctx
) {
	ASSERT(cmd_ctx != NULL);
	return cmd_ctx->bc;
}

GraphContext *CommandCtx_GetGraphContext
(
	const CommandCtx *cmd_ctx
) {
	ASSERT(cmd_ctx != NULL);
	return cmd_ctx->graph_ctx;
}

const char *CommandCtx_GetCommandName
(
	const CommandCtx *cmd_ctx
) {
	ASSERT(cmd_ctx != NULL);
	return cmd_ctx->command_name;
}

const char *CommandCtx_GetQuery
(
	const CommandCtx *cmd_ctx
) {
	ASSERT(cmd_ctx != NULL);
	return cmd_ctx->query;
}

void CommandCtx_ThreadSafeContextLock
(
	const CommandCtx *cmd_ctx
) {
	// acquire lock only when working with a blocked client
	// otherwise we're running on Redis main thread
	// no need to acquire lock
	ASSERT(cmd_ctx != NULL && cmd_ctx->ctx != NULL);
	if(cmd_ctx->bc) {
		RedisModule_ThreadSafeContextLock(cmd_ctx->ctx);
	}
}

void CommandCtx_ThreadSafeContextUnlock
(
	const CommandCtx *cmd_ctx
) {
	// release lock only when working with a blocked client
	// otherwise we're running on Redis main thread
	// no need to release lock
	ASSERT(cmd_ctx != NULL && cmd_ctx->ctx != NULL);
	if(cmd_ctx->bc) {
		RedisModule_ThreadSafeContextUnlock(cmd_ctx->ctx);
	}
}

void CommandCtx_UnblockClient
(
	CommandCtx *cmd_ctx
) {
	ASSERT(cmd_ctx != NULL);
	if(cmd_ctx->bc) {
		RedisGraph_UnblockClient(cmd_ctx->bc);
		cmd_ctx->bc = NULL;
		if(cmd_ctx->ctx) {
			RedisModule_FreeThreadSafeContext(cmd_ctx->ctx);
			cmd_ctx->ctx = NULL;
		}
	}
}

void CommandCtx_Free
(
	CommandCtx *cmd_ctx
) {
	// decrement reference count
	int prev_ref = atomic_fetch_sub (&cmd_ctx->ref_count, 1) ;
	ASSERT (prev_ref < 3) ;

	if (prev_ref == 1) {
		// reference count is zero, free command context
		ASSERT (cmd_ctx->bc == NULL) ;

		RedisModule_FreeString (NULL, cmd_ctx->rm_query) ;
		RedisModule_FreeString (NULL, cmd_ctx->rm_command_name) ;

		if (cmd_ctx->params != NULL) {
			rm_free (cmd_ctx->params) ;
		}

		rm_free (cmd_ctx) ;
	}
}

