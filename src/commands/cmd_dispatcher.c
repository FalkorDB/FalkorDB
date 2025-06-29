/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "commands.h"
#include "cmd_context.h"
#include "../util/thpool/pool.h"
#include "../util/simple_timer.h"
#include "../util/blocked_client.h"
#include "../configuration/config.h"

#define GRAPH_VERSION_MISSING -1

// command handler function pointer
typedef void(*Command_Handler)(void *args);

// read configuration flags
// returning REDIS_MODULE_ERR if flag parsing failed
static int _read_flags
(
	RedisModuleString **argv,   // commands arguments
  	int argc,                   // number of arguments
  	bool *compact,              // compact result-set format
	long long *timeout,         // query level timeout
  	bool *timeout_rw,           // apply timeout on both read and write queries
  	uint *graph_version,        // graph version [UNUSED]
  	char **errmsg,              // reported error message
	bolt_client_t **bolt_client // BOLT client
) {
	ASSERT(compact != NULL);
	ASSERT(timeout != NULL);

	long long max_timeout;

	// set defaults
	*compact       = false;  // verbose
	*bolt_client   = NULL;
	*graph_version = GRAPH_VERSION_MISSING;
	Config_Option_get(Config_TIMEOUT_DEFAULT, timeout);
	Config_Option_get(Config_TIMEOUT_MAX, &max_timeout);

	if(max_timeout != CONFIG_TIMEOUT_NO_TIMEOUT ||
	   *timeout != CONFIG_TIMEOUT_NO_TIMEOUT) {
		*timeout_rw = true;
		if(*timeout == CONFIG_TIMEOUT_NO_TIMEOUT) {
			*timeout = max_timeout;
		}
	} else {
		Config_Option_get(Config_TIMEOUT, timeout);
		*timeout_rw = false;
	}

	// GRAPH.QUERY <GRAPH_KEY> <QUERY>
	// make sure we've got more than 3 arguments
	if(argc <= 3) return REDISMODULE_OK;

	// scan arguments
	for(int i = 3; i < argc; i++) {
		const char *arg = RedisModule_StringPtrLen(argv[i], NULL);

		if(!strcasecmp(arg, "--compact")) {
			// compact result-set
			*compact = true;
		} else if(!strcasecmp(arg, "--bolt")) {
			*bolt_client = (bolt_client_t *)argv[++i];
		} else if(!strcasecmp(arg, "timeout")) {
			// query timeout
			int err = REDISMODULE_ERR;
			if(i < argc - 1) {
				i++; // Set the current argument to the timeout value.
				err = RedisModule_StringToLongLong(argv[i], timeout);

				if(max_timeout != CONFIG_TIMEOUT_NO_TIMEOUT &&
				   *timeout > max_timeout) {
					int rc __attribute__((unused));
					rc = asprintf(errmsg, "The query TIMEOUT parameter value cannot exceed the TIMEOUT_MAX configuration parameter value");
					return REDISMODULE_ERR;
				}

				if(*timeout == CONFIG_TIMEOUT_NO_TIMEOUT && timeout_rw) {
					Config_Option_get(Config_TIMEOUT_DEFAULT, timeout);
					if(*timeout == CONFIG_TIMEOUT_NO_TIMEOUT) {
						*timeout = max_timeout;
					}
				}
			}

			// Emit error on missing, negative, or non-numeric timeout values.
			if(err != REDISMODULE_OK || *timeout < 0) {
				int rc __attribute__((unused));
				rc = asprintf(errmsg, "Failed to parse query timeout value");
				return REDISMODULE_ERR;
			}
		} else if(!strcasecmp(arg, "version")) {
			long long v = GRAPH_VERSION_MISSING;
			int err = REDISMODULE_ERR;
			if(i < argc - 1) {
				i++; // Set the current argument to the version value.
				err = RedisModule_StringToLongLong(argv[i], &v);
				*graph_version = v;
			}

			// Emit error on missing, negative, or non-numeric version values.
			if(err != REDISMODULE_OK || v < 0 || v > UINT_MAX) {
				int rc __attribute__((unused));
				rc = asprintf(errmsg, "Failed to parse graph version value");
				return REDISMODULE_ERR;
			}

			continue;
		}
	}
	return REDISMODULE_OK;
}

// Returns false if client provided a graph version
// which mismatch the current graph version
static bool _verifyGraphVersion(GraphContext *gc, uint version) {
	// caller did not specify graph version
	if(version == GRAPH_VERSION_MISSING) return true;
	return (GraphContext_GetVersion(gc) == version);
}

static void _rejectOnVersionMismatch(RedisModuleCtx *ctx, uint version) {
	RedisModule_ReplyWithArray(ctx, 2);
	RedisModule_ReplyWithError(ctx, "version mismatch");
	RedisModule_ReplyWithLongLong(ctx, version);
}

// Return true if the command has a valid number of arguments.
static inline bool _validate_command_arity(GRAPH_Commands cmd, int arity) {
	switch(cmd) {
		case CMD_QUERY:
		case CMD_RO_QUERY:
		case CMD_EXPLAIN:
		case CMD_PROFILE:
			// Expect a command, graph name, a query, and optional config flags.
			return arity >= 3 && arity <= 8;
		default:
			ASSERT("encountered unhandled query type" && false);
			return false;
	}
}

// Get command handler.
static Command_Handler get_command_handler(GRAPH_Commands cmd) {
	switch(cmd) {
		case CMD_QUERY:
		case CMD_RO_QUERY:
			return Graph_Query;
		case CMD_EXPLAIN:
			return Graph_Explain;
		case CMD_PROFILE:
			return Graph_Profile;
		default:
			ASSERT(false);
	}
	return NULL;
}

static bool should_command_create_graph(GRAPH_Commands cmd) {
	switch(cmd) {
		case CMD_QUERY:
		case CMD_PROFILE:
			return true;
		case CMD_EXPLAIN:
		case CMD_RO_QUERY:
			return false;
		default:
			ASSERT(false);
	}
	return false;
}

int CommandDispatch
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	char *errmsg;
	uint version;
	bolt_client_t *bolt_client;
	bool compact;
	bool timeout_rw;
	long long timeout;
	simple_timer_t timer;
	CommandCtx *context = NULL;

	simple_tic(timer);
	uint64_t received_ts = unix_timestamp();

	RedisModuleString *graph_name = argv[1];
	RedisModuleString *query = (argc > 2) ? argv[2] : NULL;
	const char *command_name = RedisModule_StringPtrLen(argv[0], NULL);
	GRAPH_Commands cmd = CommandFromString(command_name);

	if(_validate_command_arity(cmd, argc) == false) return RedisModule_WrongArity(ctx);

	// parse additional arguments
	int res = _read_flags(argv, argc, &compact, &timeout, &timeout_rw, &version,
			&errmsg, &bolt_client);
	if(res == REDISMODULE_ERR) {
		// emit error and exit if argument parsing failed
		RedisModule_ReplyWithError(ctx, errmsg);
		free(errmsg);
		return REDISMODULE_OK;
	}

	bool shouldCreate = should_command_create_graph(cmd);
	GraphContext *gc = GraphContext_Retrieve(ctx, graph_name, true,
			shouldCreate);

	// if GraphContext is null, key access failed and an error been emitted
	if(!gc) return REDISMODULE_ERR;

	// return incase caller provided a mismatched graph version
	if(!_verifyGraphVersion(gc, version)) {
		_rejectOnVersionMismatch(ctx, GraphContext_GetVersion(gc));
		// Release the GraphContext, as we increased its reference count
		// when retrieving it.
		GraphContext_DecreaseRefCount(gc);
		return REDISMODULE_OK;
	}


	// determine the query execution context
	// queries issued within a LUA script or multi exec block must
	// run on Redis main thread, others can run on different threads
	int flags = RedisModule_GetContextFlags(ctx);
	bool is_replicated = RedisModule_GetContextFlags(ctx) &
		REDISMODULE_CTX_FLAGS_REPLICATED;

	bool main_thread = (is_replicated ||
		(flags & (REDISMODULE_CTX_FLAGS_MULTI       |
				REDISMODULE_CTX_FLAGS_LUA           |
				REDISMODULE_CTX_FLAGS_DENY_BLOCKING |
				REDISMODULE_CTX_FLAGS_LOADING)));
	ExecutorThread exec_thread = (main_thread)
		? EXEC_THREAD_MAIN
		: EXEC_THREAD_READER;

	Command_Handler handler = get_command_handler(cmd);
	if(exec_thread == EXEC_THREAD_MAIN) {
		// run query on Redis main thread
		context = CommandCtx_New(ctx, NULL, argv[0], query, gc, exec_thread,
								 is_replicated, compact, timeout, timeout_rw,
								 received_ts, timer, bolt_client);
		handler(context);
	} else {
		// run query on a dedicated thread
		RedisModuleBlockedClient *bc = bolt_client != NULL ? NULL : RedisGraph_BlockClient(ctx);
		RedisModuleCtx*redis_ctx = bolt_client != NULL ? bolt_client->ctx : NULL;
		context = CommandCtx_New(redis_ctx, bc, argv[0], query, gc, exec_thread,
								 is_replicated, compact, timeout, timeout_rw,
								 received_ts, timer, bolt_client);

		if(ThreadPool_AddWork(handler, context, false) == THPOOL_QUEUE_FULL) {
			// report an error once our workers thread pool internal queue
			// is full, this error usually happens when the server is
			// under heavy load and is unable to catch up
			RedisModule_ReplyWithError(ctx, "Max pending queries exceeded");
			// release the GraphContext, as we increased its reference count
			// when retrieving it
			GraphContext_DecreaseRefCount(gc);
			CommandCtx_UnblockClient(context);
			CommandCtx_Free(context);
		}
	}

	return REDISMODULE_OK;
}

