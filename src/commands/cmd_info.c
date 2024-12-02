/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "commands.h"
#include "../globals.h"
#include "redismodule.h"
#include "cmd_context.h"
#include "../util/thpool/pools.h"

#include <ctype.h>
#include <string.h>

#define QUERY_KEY_NAME              "Query"
#define GRAPH_NAME_KEY_NAME         "Graph name"
#define REPLICATION_KEY_NAME        "Replicated command"
#define WAIT_DURATION_KEY_NAME      "Wait duration"
#define RECEIVED_TIMESTAMP_KEY_NAME "Received at"
#define EXECUTION_DURATION_KEY_NAME "Execution duration"

#define SUBCOMMAND_NAME_RUNNING_QUERIES "RunningQueries"
#define SUBCOMMAND_NAME_WAITING_QUERIES "WaitingQueries"

//------------------------------------------------------------------------------
// Info section API
//------------------------------------------------------------------------------

// adds a new section to the info output
static void Info_AddSection
(
	RedisModuleCtx *ctx,  // redis module context
	const char *section,  // section name
	uint32_t n_entries    // number of entries
) {
	// validate arguments
	ASSERT(ctx != NULL);
	ASSERT(section != NULL);

	RedisModule_ReplyWithCString(ctx, section);
	RedisModule_ReplyWithArray(ctx, n_entries);
}

// adds a new entry to the current section
static void Info_SectionAddEntryString
(
	RedisModuleCtx *ctx,  // redis module context
	const char *entry_name,  // entry name
	const char *entry_value  // entry value
) {
	// validate arguments
	ASSERT(ctx != NULL);
	ASSERT(entry_name != NULL);
	ASSERT(entry_value != NULL);

	RedisModule_ReplyWithCString(ctx, entry_name);
	RedisModule_ReplyWithCString(ctx, entry_value);
}

// adds a new entry to the current section
static void Info_SectionAddEntryLongLong
(
	RedisModuleCtx *ctx,  // redis module context
	const char *entry_name,  // entry name
	long long entry_value    // entry value
) {
	// validate arguments
	ASSERT(ctx != NULL);
	ASSERT(entry_name != NULL);

	RedisModule_ReplyWithCString(ctx, entry_name);
	RedisModule_ReplyWithLongLong(ctx, entry_value);
}

// adds a new entry to the current section
static void Info_SectionAddEntryDouble
(
	RedisModuleCtx *ctx,  // redis module context
	const char *entry_name,  // entry name
	double entry_value       // entry value
) {
	// validate arguments
	ASSERT(ctx != NULL);
	ASSERT(entry_name != NULL);

	RedisModule_ReplyWithCString(ctx, entry_name);
	RedisModule_ReplyWithDouble(ctx, entry_value);
}

// replies with query information
static void _emit_running_query
(
    RedisModuleCtx *ctx,   // redis module context
    const CommandCtx *cmd  // command context
) {
	ASSERT(ctx != NULL);
	ASSERT(cmd != NULL);

	// compute query execution time
	const double total_time = TIMER_GET_ELAPSED_MILLISECONDS(cmd->timer);

	RedisModule_ReplyWithArray(ctx, 5 * 2);

	// emit query received timestamp
	Info_SectionAddEntryLongLong(ctx, RECEIVED_TIMESTAMP_KEY_NAME,
			cmd->received_ts);

	// emit graph name
	Info_SectionAddEntryString(ctx, GRAPH_NAME_KEY_NAME,
			GraphContext_GetName(cmd->graph_ctx));

	// emit query
	Info_SectionAddEntryString(ctx, QUERY_KEY_NAME, cmd->query);

	// emit query wait duration
	Info_SectionAddEntryDouble(ctx, EXECUTION_DURATION_KEY_NAME, total_time);

	// emit rather or not query was replicated
	Info_SectionAddEntryLongLong(ctx, REPLICATION_KEY_NAME,
			cmd->replicated_command);
}

// replies with query information
static void _emit_waiting_query
(
	RedisModuleCtx *ctx,   // redis module context
	const CommandCtx *cmd  // command context
) {
	ASSERT(ctx != NULL);
	ASSERT(cmd != NULL);

	// compute query execution time
	const double total_time = TIMER_GET_ELAPSED_MILLISECONDS(cmd->timer);

	RedisModule_ReplyWithArray(ctx, 4 * 2);

	// emit query received timestamp
	Info_SectionAddEntryLongLong(ctx, RECEIVED_TIMESTAMP_KEY_NAME,
			cmd->received_ts);

	// emit graph name
	Info_SectionAddEntryString(ctx, GRAPH_NAME_KEY_NAME,
			GraphContext_GetName(cmd->graph_ctx));

	// emit query
	Info_SectionAddEntryString(ctx, QUERY_KEY_NAME, cmd->query);

	// emit query execution duration
	Info_SectionAddEntryDouble(ctx, WAIT_DURATION_KEY_NAME, total_time);
}

// handles the "GRAPH.INFO RunningQueries" section
// "GRAPH.INFO RunningQueries"
static void _info_running_queries
(
	RedisModuleCtx *ctx       // redis context
) {
	// an example for a command and reply:
	// command:
	// GRAPH.INFO RunningQueries
	// reply:
	// "RunningQueries"
	//     "Received at"
	//     "Graph name"
	//     "Query"
	//     "Execution duration"
	//     "Replicated command"

	ASSERT(ctx != NULL);

	//--------------------------------------------------------------------------
	// collect running queries
	//--------------------------------------------------------------------------

	// get all currently executing commands
	// #readers + #writers + Redis main thread
	uint32_t n = ThreadPools_ThreadCount() + 1;
	CommandCtx* cmds[n];
	Globals_GetCommandCtxs(cmds, &n);

	// create a new subsection in the reply
	Info_AddSection(ctx, "# Running queries", n);

	for(uint32_t i = 0; i < n; i++) {
		CommandCtx *cmd = cmds[i];
		ASSERT(cmd != NULL);

		// emit the command
		_emit_running_query(ctx, cmd);

		// decrease the command's ref count
		// free the command if it's no longer referenced
		CommandCtx_Free(cmd);
	}
}

// handles the "GRAPH.INFO WaitingQueries" section
// "GRAPH.INFO WaitingQueries"
static void _info_waiting_queries
(
	RedisModuleCtx *ctx       // redis context
) {
	// an example for a command and reply:
	// command:
	// GRAPH.INFO WaitingQueries
	// reply:
	// "WaitingQueries"
	//     "Received at"
	//     "Graph name"
	//     "Query"
	//     "Wait duration"

	ASSERT(ctx != NULL);

	//--------------------------------------------------------------------------
	// collect waiting queries
	//--------------------------------------------------------------------------

	uint32_t n;
	CommandCtx **cmds = (CommandCtx**)ThreadPools_GetTasksByHandler(Graph_Query,
			(void(*)(void *))CommandCtx_Incref, &n);

	// create a new subsection in the reply
	Info_AddSection(ctx, "# Waiting queries", n);

	for(uint32_t i = 0; i < n; i++) {
		CommandCtx *cmd = cmds[i];
		ASSERT(cmd != NULL);

		// emit the command
		_emit_waiting_query(ctx, cmd);

		// decrease the command's ref count
		// free the command if it's no longer referenced
		CommandCtx_Free(cmd);
	}

	free(cmds);
}

// attempts to find the specified sections of "GRAPH.INFO" and dispatch it
static void _handle_sections
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // command arguments
	const int argc             // number of arguments
) {
	ASSERT(ctx    != NULL);
	ASSERT(argv   != NULL);

	int section_count = 0;
	bool running_queries = false;
	bool waiting_queries = false;

	if(argc == 0) {
		running_queries = true;
		waiting_queries = true;
		section_count = 2;
	} else {
		for(uint i = 0; i < argc; i++) {
			const char *subcmd = RedisModule_StringPtrLen(argv[i], NULL);
			if(!running_queries &&
			   !strcasecmp(subcmd, SUBCOMMAND_NAME_RUNNING_QUERIES)) {
				running_queries = true;
				section_count++;
			} else if(!waiting_queries &&
					  !strcasecmp(subcmd, SUBCOMMAND_NAME_WAITING_QUERIES)) {
				waiting_queries = true;
				section_count++;
			}
		}
	}

	if(section_count == 0) {
		RedisModule_ReplyWithCString(ctx, "no section found");
		return;
	}

	RedisModule_ReplyWithArray(ctx, section_count * 2);
	if(running_queries) {
		_info_running_queries(ctx);
	}
	if(waiting_queries) {
		_info_waiting_queries(ctx);
	}
}

// graph.info command handler
// GRAPH.INFO [Section [Section ...]]
// GRAPH.INFO RunningQueries WaitingQueries
int Graph_Info
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // command arguments
	const int argc             // number of arguments
) {
	ASSERT(ctx != NULL);

	// expecting at least two arguments
	if (argc < 1) {
		return RedisModule_WrongArity(ctx);
	}

	_handle_sections(ctx, argv + 1, argc - 1);

	return REDISMODULE_OK;
}

