/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "../ast/ast.h"
#include "cmd_context.h"
#include "../util/arr.h"
#include "cron/cron.h"
#include "../globals.h"
#include "../query_ctx.h"
#include "execution_ctx.h"
#include "../graph/graph.h"
#include "../util/rmalloc.h"
#include "../errors/errors.h"
#include "../index/indexer.h"
#include "../effects/effects.h"
#include "../util/cache/cache.h"
#include "../util/thpool/pools.h"
#include "../configuration/config.h"
#include "../execution_plan/execution_plan.h"

// GraphQueryCtx stores the allocations required to execute a query.
typedef struct {
	GraphContext *graph_ctx;  // graph context
	RedisModuleCtx *rm_ctx;   // redismodule context
	QueryCtx *query_ctx;      // query context
	ExecutionCtx *exec_ctx;   // execution context
	CommandCtx *command_ctx;  // command context
	CronTaskHandle timeout;   // timeout cron task
} GraphQueryCtx;

static GraphQueryCtx *GraphQueryCtx_New
(
	GraphContext *graph_ctx,
	RedisModuleCtx *rm_ctx,
	ExecutionCtx *exec_ctx,
	CommandCtx *command_ctx,
	QueryExecutionTypeFlag flags,
	CronTaskHandle timeout
) {
	GraphQueryCtx *ctx = rm_malloc(sizeof(GraphQueryCtx));

	ctx->rm_ctx           =  rm_ctx;
	ctx->exec_ctx         =  exec_ctx;
	ctx->graph_ctx        =  graph_ctx;
	ctx->query_ctx        =  QueryCtx_GetQueryCtx();
	ctx->query_ctx->flags = flags;
	ctx->command_ctx      =  command_ctx;
	ctx->timeout          =  timeout;

	return ctx;
}

static void inline GraphQueryCtx_Free
(
	GraphQueryCtx *ctx
) {
	ASSERT(ctx != NULL);
	rm_free(ctx);
}

// checks if we should replicate command to replicas via
// the original GRAPH.QUERY command
// or via a set of effects GRAPH.EFFECT
// we would prefer to use effects when the number of effects is relatively small
// compared to query's execution time
static bool _should_replicate_effects(void)
{
	// GRAPH.EFFECT will be used to replicate a query when
	// the average modification time > configuted replicate effects threshold
	//
	// for example:
	// a query which ran for 10ms and performed 5 changes
	// the average change time is 10/5 = 2ms
	// if 2ms > configured replicate effects threshold
	// then the query will be replicated via GRAPH.EFFECT
	//
	// on the other hand if a query ran for 1ms and performed 4 changes
	// the average change timw is 1/4 = 0.25ms
	// 0.25 < configured replicate effects threshold
	// then the query will be replicate via GRAPH.QUERY

	//--------------------------------------------------------------------------
	// consult with configuration
	//--------------------------------------------------------------------------

	uint64_t effects_threshold;
	Config_Option_get(Config_EFFECTS_THRESHOLD, &effects_threshold);

	// compute average change time:
	// avg modification time = query execution time / #modifications
	double exec_time = QueryCtx_GetRuntime();
	uint64_t n = EffectsBuffer_Length(QueryCtx_GetEffectsBuffer());
	ASSERT(n > 0);
	double avg_mod_time = exec_time / n;

	avg_mod_time *= 1000; // convert from ms to μs microseconds

	// use GRAPH.EFFECT when avg_mod_time > effects_threshold
	return (avg_mod_time > (double)effects_threshold);
}

static bool abort_and_check_timeout
(
	GraphQueryCtx *gq_ctx,
	ExecutionPlan *plan
) {
	// abort timeout if set
	if(gq_ctx->timeout != 0) {
		Cron_AbortTask(gq_ctx->timeout);
	}

	// emit error if query timed out
	// TODO for the abort feature this won't necessarily mean a timeout,
	// it will also flag the aborted queries.
	const bool has_timed_out = ExecutionPlan_Drained(plan);
	if (has_timed_out) {
		ErrorCtx_SetError(EMSG_QUERY_TIMEOUT);
	}

	return has_timed_out;
}

static bool _index_operation_delete
(
	GraphContext *gc,
	AST *ast
) {
	Schema *s = NULL;
	SchemaType schema_type = SCHEMA_NODE;
	const cypher_astnode_t *index_op = ast->root;

	// retrieve strings from AST node
	const char *label = cypher_ast_label_get_name(
			cypher_ast_drop_props_index_get_label(index_op));
	const char *attr = cypher_ast_prop_name_get_value(
			cypher_ast_drop_props_index_get_prop_name(index_op, 0));

	Attribute_ID attr_id = GraphContext_GetAttributeID(gc, attr);

	// try deleting a NODE EXACT-MATCH index

	// lock
	QueryCtx_LockForCommit();

	s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
	if(s != NULL) {
		if(Schema_GetIndex(s, &attr_id, 1, IDX_EXACT_MATCH, true) != NULL) {
			// try deleting an exact match node index
			return GraphContext_DeleteIndex(gc, SCHEMA_NODE, label, attr, IDX_EXACT_MATCH);
		}
	}

	// try removing from an edge schema
	s = GraphContext_GetSchema(gc, label, SCHEMA_EDGE);
	if(s != NULL) {
		if(Schema_GetIndex(s, &attr_id, 1, IDX_EXACT_MATCH, true) != NULL) {
			// try deleting an exact match edge index
			return GraphContext_DeleteIndex(gc, SCHEMA_EDGE, label, attr, IDX_EXACT_MATCH);
		}
	}

	// no matching index
	ErrorCtx_SetError(EMSG_UNABLE_TO_DROP_INDEX, label, attr);

	return false;
}

// create index structure
static void _index_operation_create
(
	RedisModuleCtx *ctx,
	GraphContext *gc,
	AST *ast
) {
	ASSERT(gc  != NULL);
	ASSERT(ctx != NULL);
	ASSERT(ast != NULL);

	uint nprops            = 0;            // number of fields indexed
	const char *label      = NULL;         // label being indexed
	SchemaType schema_type = SCHEMA_NODE;  // type of entities being indexed

	const cypher_astnode_t *index_op = ast->root;
	cypher_astnode_type_t t = cypher_astnode_type(index_op);

	//--------------------------------------------------------------------------
	// retrieve label and attributes from AST
	//--------------------------------------------------------------------------

	if(t == CYPHER_AST_CREATE_NODE_PROPS_INDEX) {
		// old format
		// CREATE INDEX ON :N(name)
		nprops = cypher_ast_create_node_props_index_nprops(index_op);
		label  = cypher_ast_label_get_name(
				cypher_ast_create_node_props_index_get_label(index_op));
	} else {
		// new format
		// CREATE INDEX FOR (n:N) ON n.name
		nprops = cypher_ast_create_pattern_props_index_nprops(index_op);
		label  = cypher_ast_label_get_name(
				cypher_ast_create_pattern_props_index_get_label(index_op));

		// determine if index is created over node label or edge relationship
		// default to node
		if(cypher_ast_create_pattern_props_index_pattern_is_relation(index_op)) {
			schema_type = SCHEMA_EDGE;
		}
	}

	ASSERT(nprops > 0);
	ASSERT(label != NULL);

	const char *fields[nprops];
	for(uint i = 0; i < nprops; i++) {
		const cypher_astnode_t *prop_name =
			(t == CYPHER_AST_CREATE_NODE_PROPS_INDEX) ?
			cypher_ast_create_node_props_index_get_prop_name(index_op, i) :
			cypher_ast_property_operator_get_prop_name
			(cypher_ast_create_pattern_props_index_get_property_operator(index_op, i));

		fields[i] = cypher_ast_prop_name_get_value(prop_name);
	}

	// lock
	QueryCtx_LockForCommit();

	Index idx;
	// add fields to index
	if(GraphContext_AddExactMatchIndex(&idx, gc, schema_type, label, fields,
				nprops, true)) {
		Schema *s = GraphContext_GetSchema(gc, label, schema_type);
		Indexer_PopulateIndex(gc, s, idx);
	}
}

// handle index/constraint operation
// either index/constraint creation or index/constraint deletion
static void _index_operation
(
	RedisModuleCtx *ctx,
	GraphContext *gc,
	AST *ast,
	ExecutionType exec_type
) {
	switch(exec_type) {
		case EXECUTION_TYPE_INDEX_CREATE:
			_index_operation_create(ctx, gc, ast);
			break;
		case EXECUTION_TYPE_INDEX_DROP:
			_index_operation_delete(gc, ast);
			break;
		default:
			ErrorCtx_SetError(EMSG_UNKNOWN_EXECUTION_TYPE);
	}
}

//------------------------------------------------------------------------------
// Query timeout
//------------------------------------------------------------------------------

// timeout handler
void QueryTimedOut(void *pdata) {
	ASSERT(pdata != NULL);
	ExecutionPlan *plan = (ExecutionPlan *)pdata;
	ExecutionPlan_Drain(plan);
}

// set timeout for query execution
CronTaskHandle Query_SetTimeOut(uint timeout, ExecutionPlan *plan) {
	// increase execution plan ref count
	return Cron_AddTask(timeout, QueryTimedOut, NULL, plan);
}

inline static bool _readonly_cmd_mode(CommandCtx *ctx) {
	return strcasecmp(CommandCtx_GetCommandName(ctx), "graph.RO_QUERY") == 0;
}

// _ExecuteQuery accepts a GraphQueryCtx as an argument
// it may be called directly by a reader thread or the Redis main thread,
// or dispatched as a worker thread job when used for writing.
static void _ExecuteQuery(void *args) {
	ASSERT(args != NULL);

	GraphQueryCtx  *gq_ctx      = args;
	QueryCtx       *query_ctx   = gq_ctx->query_ctx;
	GraphContext   *gc          = gq_ctx->graph_ctx;
	RedisModuleCtx *rm_ctx      = gq_ctx->rm_ctx;
	ExecutionCtx   *exec_ctx    = gq_ctx->exec_ctx;
	CommandCtx     *command_ctx = gq_ctx->command_ctx;
	AST            *ast         = exec_ctx->ast;
	ExecutionPlan  *plan        = exec_ctx->plan;
	ExecutionType  exec_type    = exec_ctx->exec_type;
	const bool     profile      = (query_ctx->flags & QueryExecutionTypeFlag_PROFILE);
	const bool     readonly     = !(query_ctx->flags & QueryExecutionTypeFlag_WRITE);

	// if we have migrated to a writer thread,
	// update thread-local storage and track the CommandCtx
	if (command_ctx->thread == EXEC_THREAD_WRITER) {
		// transition the query from waiting to executing
		QueryCtx_AdvanceStage(query_ctx);
		QueryCtx_SetTLS(query_ctx);
		Globals_TrackCommandCtx(command_ctx);
	}

	// instantiate the query ResultSet
	bool compact = command_ctx->compact;
	// replicated command don't need to return result
	ResultSetFormatterType resultset_format =
		profile || command_ctx->replicated_command
		? FORMATTER_NOP
		: (compact)
			? FORMATTER_COMPACT
			: FORMATTER_VERBOSE;
	ResultSet *result_set = NewResultSet(rm_ctx, resultset_format);
	if(exec_ctx->cached) {
		ResultSet_CachedExecution(result_set); // indicate a cached execution
	}

	QueryCtx_SetResultSet(result_set);

	// acquire the appropriate lock
	if(readonly) {
		Graph_AcquireReadLock(gc->g);
	} else {
		// if this is a writer query `we need to re-open the graph key with write flag
		// this notifies Redis that the key is "dirty" any watcher on that key will
		// be notified
		CommandCtx_ThreadSafeContextLock(command_ctx);
		{
			GraphContext_MarkWriter(rm_ctx, gc);
		}
		CommandCtx_ThreadSafeContextUnlock(command_ctx);
	}

	if(exec_type == EXECUTION_TYPE_QUERY) {  // query operation
		// set policy after lock acquisition,
		// avoid resetting policies between readers and writers
		Graph_SetMatrixPolicy(gc->g, SYNC_POLICY_FLUSH_RESIZE);

		ExecutionPlan_PreparePlan(plan);
		if(profile) {
			ExecutionPlan_Profile(plan);
			if (abort_and_check_timeout(gq_ctx, plan)) {
				query_ctx->status = QueryExecutionStatus_TIMEDOUT;
			}

			if(!ErrorCtx_EncounteredError()) {
				// transition the query from executing reporting
				QueryCtx_AdvanceStage(query_ctx);
				ExecutionPlan_Print(plan, rm_ctx);
			}
		}
		else {
			result_set = ExecutionPlan_Execute(plan);
			if (abort_and_check_timeout(gq_ctx, plan)) {
				query_ctx->status = QueryExecutionStatus_TIMEDOUT;
			}
		}

		ExecutionPlan_Free(plan);
		exec_ctx->plan = NULL;
	} else if(exec_type == EXECUTION_TYPE_INDEX_CREATE ||
			exec_type == EXECUTION_TYPE_INDEX_DROP) {
		_index_operation(rm_ctx, gc, ast, exec_type);
	} else {
		ASSERT("Unhandled query type" && false);
	}

	// in case of an error, rollback any modifications
	if(ErrorCtx_EncounteredError()) {
		QueryCtx_Rollback();
		// clear resultset statistics, avoiding commnad being replicated
		ResultSet_Clear(result_set);
		if (query_ctx->status != QueryExecutionStatus_TIMEDOUT) {
			query_ctx->status = QueryExecutionStatus_FAILURE;
		}
	} else {
		// replicate if graph was modified
		if(ResultSetStat_IndicateModification(&result_set->stats)) {
			// determine rather or not to replicate via effects
			if(EffectsBuffer_Length(QueryCtx_GetEffectsBuffer()) > 0 &&
			   _should_replicate_effects()) {
				// compute effects buffer
				size_t effects_len = 0;
				u_char *effects = EffectsBuffer_Buffer(
						QueryCtx_GetEffectsBuffer(), &effects_len);
				ASSERT(effects_len > 0 && effects != NULL);

				// replicate effects
				RedisModule_Replicate(rm_ctx, "GRAPH.EFFECT", "cb!",
						GraphContext_GetName(gc), effects, effects_len);
				rm_free(effects);
			} else {
				// replicate original query
				QueryCtx_Replicate(query_ctx);
			}
		}	
	}

	QueryCtx_UnlockCommit();

	if(!profile || ErrorCtx_EncounteredError()) {
		// if we encountered an error, ResultSet_Reply will emit the error
		// send result-set back to client
		// transition the query from executing reporting
		QueryCtx_AdvanceStage(query_ctx);
		ResultSet_Reply(result_set);

		// transition the query from reporting to finished
		QueryCtx_AdvanceStage(query_ctx);
	}

	if(readonly) Graph_ReleaseLock(gc->g); // release read lock

	// log query to slowlog
	SlowLog *slowlog = GraphContext_GetSlowLog(gc);
	SlowLog_Add(slowlog, command_ctx->command_name, command_ctx->query,
				QueryCtx_GetRuntime(), NULL);

	// clean up
	ExecutionCtx_Free(exec_ctx);
	GraphContext_DecreaseRefCount(gc);
	Globals_UntrackCommandCtx(command_ctx);
	CommandCtx_UnblockClient(command_ctx);
	CommandCtx_Free(command_ctx);
	QueryCtx_Free(); // reset the QueryCtx and free its allocations
	ErrorCtx_Clear();
	ResultSet_Free(result_set);
	GraphQueryCtx_Free(gq_ctx);
}

static void _DelegateWriter(GraphQueryCtx *gq_ctx) {
	ASSERT(gq_ctx != NULL);

	//---------------------------------------------------------------------------
	// Migrate to writer thread
	//---------------------------------------------------------------------------

	// write queries will be executed on a dedicated writer thread,
	// clear this thread data
	ErrorCtx_Clear();
	QueryCtx_RemoveFromTLS();

	// untrack the CommandCtx
	Globals_UntrackCommandCtx(gq_ctx->command_ctx);

	// update execution thread to writer
	gq_ctx->command_ctx->thread = EXEC_THREAD_WRITER;

	// reset query stage from executing back to waiting
	QueryCtx_ResetStage(gq_ctx->query_ctx);

	// dispatch work to the writer thread
	int res = ThreadPools_AddWorkWriter(_ExecuteQuery, gq_ctx, 0);
	ASSERT(res == 0);
}

void _query
(
	bool profile,
	void *args
) {
	CommandCtx     *command_ctx = (CommandCtx *)args;
	QueryCtx       *query_ctx   = QueryCtx_GetQueryCtx();
	RedisModuleCtx *ctx         = CommandCtx_GetRedisCtx(command_ctx);
	GraphContext   *gc          = CommandCtx_GetGraphContext(command_ctx);
	ExecutionCtx   *exec_ctx    = NULL;

	Globals_TrackCommandCtx(command_ctx);
	QueryCtx_SetGlobalExecutionCtx(command_ctx);

	// transition the query from waiting to executing
	QueryCtx_AdvanceStage(query_ctx);

	// parse query parameters and build an execution plan
	// or retrieve it from the cache
	exec_ctx = ExecutionCtx_FromQuery(command_ctx->query);
	if(exec_ctx == NULL) goto cleanup;

	// update cached flag
	QueryCtx_SetUtilizedCache(query_ctx, exec_ctx->cached);

	ExecutionType exec_type = exec_ctx->exec_type;
	bool readonly = AST_ReadOnly(exec_ctx->ast->root);
	bool index_op = (exec_type == EXECUTION_TYPE_INDEX_CREATE ||
	     exec_type == EXECUTION_TYPE_INDEX_DROP);

	if(profile && index_op) {
		RedisModule_ReplyWithError(ctx, "Can't profile index operations.");
		goto cleanup;
	}

	// write query executing via GRAPH.RO_QUERY isn't allowed
	if(!profile && !readonly && _readonly_cmd_mode(command_ctx)) {
		ErrorCtx_SetError(EMSG_MISUSE_GRAPH_ROQUERY);
		goto cleanup;
	}

	CronTaskHandle timeout_task = 0;

	// enforce specified timeout when query is readonly
	// or timeout applies to both read and write
	bool enforce_timeout = command_ctx->timeout != 0 && !index_op &&
		(readonly || command_ctx->timeout_rw) &&
		!command_ctx->replicated_command;
	if(enforce_timeout) {
		timeout_task = Query_SetTimeOut(command_ctx->timeout, exec_ctx->plan);
	}

	// populate the container struct for invoking _ExecuteQuery.
	QueryExecutionTypeFlag flags = QueryExecutionTypeFlag_READ;
	if (!readonly) {
		flags |= QueryExecutionTypeFlag_WRITE;
	}
	if (profile) {
		flags |= QueryExecutionTypeFlag_PROFILE;
	}
	GraphQueryCtx *gq_ctx = GraphQueryCtx_New(gc, ctx, exec_ctx, command_ctx,
											  flags, timeout_task);

	// if 'thread' is redis main thread, continue running
	// if readonly is true we're executing on a worker thread from
	// the read-only threadpool
	if(readonly || command_ctx->thread == EXEC_THREAD_MAIN) {
		_ExecuteQuery(gq_ctx);
	} else {
		_DelegateWriter(gq_ctx);
	}

	return;

cleanup:
	// if there were any query compile time errors, report them
	if(ErrorCtx_EncounteredError()) {
		ErrorCtx_EmitException();
	}

	// Cleanup routine invoked after encountering errors in this function.
	ExecutionCtx_Free(exec_ctx);
	GraphContext_DecreaseRefCount(gc);
	Globals_UntrackCommandCtx(command_ctx);
	CommandCtx_UnblockClient(command_ctx);
	CommandCtx_Free(command_ctx);
	QueryCtx_Free(); // Reset the QueryCtx and free its allocations.
	ErrorCtx_Clear();
}

void Graph_Profile(void *args) {
	_query(true, args);
}

void Graph_Query(void *args) {
	_query(false, args);
}

