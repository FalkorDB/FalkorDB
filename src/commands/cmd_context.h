/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "cypher-parser.h"
#include "../redismodule.h"
#include "../util/simple_timer.h"
#include "../graph/graphcontext.h"
#include "../bolt/bolt.h"

#include <stdatomic.h>

// ExecutorThread lists the diffrent types of threads in the system
typedef enum {
	EXEC_THREAD_MAIN,    // redis main thread
	EXEC_THREAD_READER,  // read only thread
	EXEC_THREAD_WRITER,  // write only thread
} ExecutorThread;

// command context, used for concurrent query processing
typedef struct {
	char *query;                   // query string
	RedisModuleCtx *ctx;           // redis module context
	char *command_name;            // command to execute
	GraphContext *graph_ctx;       // graph context
	atomic_int ref_count;          // reference count
	RedisModuleBlockedClient *bc;  // blocked client
	bool replicated_command;       // whether this instance was spawned by a replication command
	bool compact;                  // whether this query was issued with the compact flag
	ExecutorThread thread;         // which thread executes this command
	long long timeout;             // the query timeout, if specified
	bool timeout_rw;               // apply timeout on both read and write queries
	uint64_t received_ts;          // command received at this UNIX timestamp
	simple_timer_t timer;          // stopwatch started upon command received
	bolt_client_t *bolt_client;    // BOLT client
} CommandCtx;

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
	uint64_t received_ts,          // command received at this  UNIX timestamp
	simple_timer_t timer,          // stopwatch started upon command received
	bolt_client_t *bolt_client     // BOLT client
);

// increment command context reference count
void CommandCtx_Incref
(
	CommandCtx *command_ctx
);

// get Redis module context
RedisModuleCtx *CommandCtx_GetRedisCtx
(
	CommandCtx *command_ctx
);

// get BOLT client
bolt_client_t *CommandCtx_GetBoltClient
(
	CommandCtx *command_ctx
);

// get blocking client
RedisModuleBlockedClient *CommandCtx_GetBlockingClient
(
	const CommandCtx *command_ctx
);

// get GraphContext
GraphContext *CommandCtx_GetGraphContext
(
	const CommandCtx *command_ctx
);

// get command name
const char *CommandCtx_GetCommandName
(
	const CommandCtx *command_ctx
);

// get query string
const char *CommandCtx_GetQuery
(
	const CommandCtx *command_ctx
);

// acquire Redis global lock
void CommandCtx_ThreadSafeContextLock
(
	const CommandCtx *command_ctx
);

// release Redis global lock
void CommandCtx_ThreadSafeContextUnlock
(
	const CommandCtx *command_ctx
);

// unblock the client
void CommandCtx_UnblockClient
(
	CommandCtx *command_ctx
);

// free command context
void CommandCtx_Free
(
	CommandCtx *command_ctx
);

