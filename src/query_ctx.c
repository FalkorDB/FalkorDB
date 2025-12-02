/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "query_ctx.h"
#include "RG.h"
#include "errors.h"
#include "globals.h"
#include "util/simple_timer.h"
#include "arithmetic/arithmetic_expression.h"
#include "serializers/graphcontext_type.h"

// GraphContext type as it is registered at Redis
extern RedisModuleType *GraphContextRedisModuleType;

pthread_key_t _tlsQueryCtxKey;  // thread local storage query context key

// retrieve or instantiate new QueryCtx
static inline QueryCtx *_QueryCtx_GetCreateCtx(void) {
	QueryCtx *ctx = pthread_getspecific(_tlsQueryCtxKey);

	if(ctx == NULL) {
		// set a new thread-local QueryCtx if one has not been created
		ctx = rm_calloc(1, sizeof(QueryCtx));

		// created lazily only when needed
		ctx->undo_log       = NULL ;
		ctx->effects_buffer = NULL ;
		ctx->stage          = QueryStage_WAITING ;  // initial query stage
		ctx->deterministic  = true ;                // assuming deterministic

		pthread_setspecific(_tlsQueryCtxKey, ctx);
	}

	return ctx;
}

// retrieve QueryCtx, return NULL if one does not exist
static inline QueryCtx *_QueryCtx_GetCtx(void) {
	QueryCtx *ctx = pthread_getspecific(_tlsQueryCtxKey);
	return ctx;
}

// rax callback routine for freeing computed parameter values
static void _ParameterFreeCallback(void *param_val) {
	SIValue *val = (SIValue*)param_val;
	SIValue_Free(*val);
	rm_free(val);
}

bool QueryCtx_Init(void) {
	return (pthread_key_create(&_tlsQueryCtxKey, NULL) == 0);
}

inline QueryCtx *QueryCtx_GetQueryCtx(void) {
	return _QueryCtx_GetCreateCtx();
}

inline void QueryCtx_SetTLS
(
	QueryCtx *query_ctx
) {
	pthread_setspecific(_tlsQueryCtxKey, query_ctx);
}

inline void QueryCtx_RemoveFromTLS(void) {
	pthread_setspecific(_tlsQueryCtxKey, NULL);
}

//------------------------------------------------------------------------------
// query statistics
//------------------------------------------------------------------------------

// returns the number of milliseconds the timer has counted
// this function reset the timer
static double _QueryCtx_GetCountedMilliseconds
(
	QueryCtx *ctx
) {
	ASSERT(ctx != NULL);

	double ms = TIMER_GET_ELAPSED_MILLISECONDS(ctx->stats.timer);

	// resets the stage timer
	simple_tic(ctx->stats.timer);

	return ms;
}

// reads the stage timer and updates the current stage duration
static void _QueryCtx_UpdateStageDuration
(
	QueryCtx *ctx  // query context
) {
	ASSERT(ctx != NULL);

	// update stage duration
	ctx->stats.durations[ctx->stage] += _QueryCtx_GetCountedMilliseconds(ctx);
}

// advance query's stage
// waiting   -> executing
// executing -> reporting
// reporting -> finished
void QueryCtx_AdvanceStage
(
	QueryCtx *ctx  // query context
) {
	ASSERT(ctx != NULL);

	// validate query stage
	ASSERT(ctx->stage >= QueryStage_WAITING);
	ASSERT(ctx->stage <= QueryStage_REPORTING);

	// update stage duration
	_QueryCtx_UpdateStageDuration(ctx);

	if(ctx->stage == QueryStage_REPORTING) {
		// done reporting, log query
		GraphContext_LogQuery (ctx->gc,
				ctx->stats.received_ts,
				ctx->stats.durations[QueryStage_WAITING],
				ctx->stats.durations[QueryStage_EXECUTING],
				ctx->stats.durations[QueryStage_REPORTING],
				ctx->stats.parameterized,
				ctx->stats.utilized_cache,
				ctx->flags & QueryExecutionTypeFlag_WRITE,
				ctx->status == QueryExecutionStatus_TIMEDOUT,
				ctx->query_data.query_params_len,
				ctx->query_data.query) ;
	}

	// advance to next stage
	ctx->stage++;
}

// regress query's stage
// waiting <- executing
void QueryCtx_ResetStage
(
	QueryCtx *ctx  // query context
) {
	ASSERT(ctx != NULL);

	// a query can only transition back from executing to waiting
	ASSERT(ctx->stage == QueryStage_EXECUTING);

	// update stage duration
	_QueryCtx_UpdateStageDuration(ctx);

	// transition from executing to waiting
	ctx->stage = QueryStage_WAITING;
}

// sets the "utilized_cache" flag of a QueryInfo
void QueryCtx_SetUtilizedCache
(
    QueryCtx *ctx,  // query context
    bool utilized   // cache utilized
) {
	ASSERT(ctx != NULL);

	ctx->stats.utilized_cache = utilized;
}

// sets the global execution context
void QueryCtx_SetGlobalExecutionCtx
(
	CommandCtx *cmd_ctx
) {
	ASSERT(cmd_ctx != NULL);

	QueryCtx *ctx = _QueryCtx_GetCreateCtx();

	ctx->gc                           = CommandCtx_GetGraphContext(cmd_ctx);
	ctx->query_data.query             = CommandCtx_GetQuery(cmd_ctx);
	ctx->global_exec_ctx.bc           = CommandCtx_GetBlockingClient(cmd_ctx);
	ctx->global_exec_ctx.redis_ctx    = CommandCtx_GetRedisCtx(cmd_ctx);
	ctx->global_exec_ctx.bolt_client  = CommandCtx_GetBoltClient(cmd_ctx);
	ctx->global_exec_ctx.command_name = CommandCtx_GetCommandName(cmd_ctx);

	// copy command's timer
	simple_timer_copy(cmd_ctx->timer, ctx->stats.timer);

	// received timestamp (epoch time)
	ctx->stats.received_ts = cmd_ctx->received_ts;
}

// set the provided AST for access through the QueryCtx
void QueryCtx_SetAST
(
	AST *ast
) {
	QueryCtx *ctx = _QueryCtx_GetCreateCtx();
	ctx->query_data.ast = ast;
}

// set the provided GraphCtx for access through the QueryCtx
void QueryCtx_SetGraphCtx
(
	GraphContext *gc
) {
	ASSERT(gc != NULL);
	QueryCtx *ctx = _QueryCtx_GetCreateCtx();
	ctx->gc = gc;
}

// set the resultset
void QueryCtx_SetResultSet
(
	ResultSet *result_set
) {
	ASSERT(result_set != NULL);

	QueryCtx *ctx = _QueryCtx_GetCreateCtx();
	ctx->internal_exec_ctx.result_set = result_set;
}

// set the parameters map
void QueryCtx_SetParams
(
	dict *params
) {
	ASSERT(params != NULL);

	QueryCtx *ctx = _QueryCtx_GetCreateCtx();
	ctx->query_data.params = params;
}

// mark query context as not deterministic
void QueryCtx_SetNonDeterministic (void) {
	QueryCtx *ctx = _QueryCtx_GetCreateCtx () ;
	ctx->deterministic = false ;
}

// returns true if query is deterministic
bool QueryCtx_IsDeterministic (void) {
	QueryCtx *ctx = _QueryCtx_GetCreateCtx () ;
	return ctx->deterministic ;
}

// retrieve the AST
AST *QueryCtx_GetAST(void) {
	QueryCtx *ctx = _QueryCtx_GetCtx();
	ASSERT(ctx != NULL);
	return ctx->query_data.ast;
}

// retrieve the query parameters values map
dict *QueryCtx_GetParams(void) {
	QueryCtx *ctx = _QueryCtx_GetCtx();
	ASSERT(ctx != NULL);
	return ctx->query_data.params;
}

// retrieve the GraphCtx
GraphContext *QueryCtx_GetGraphCtx(void) {
	QueryCtx *ctx = _QueryCtx_GetCtx();
	ASSERT(ctx && ctx->gc);
	return ctx->gc;
}

// retrieve the Graph object
Graph *QueryCtx_GetGraph(void) {
	GraphContext *gc = QueryCtx_GetGraphCtx();
	return gc->g;
}

// retrieve undo log
UndoLog QueryCtx_GetUndoLog(void) {
	QueryCtx *ctx = _QueryCtx_GetCtx();
	ASSERT(ctx != NULL);
	
	if(ctx->undo_log == NULL) {
		ctx->undo_log = UndoLog_New();
	}
	return ctx->undo_log;
}

// rollback the current command
void QueryCtx_Rollback(void) {
	QueryCtx *ctx = _QueryCtx_GetCtx();
	ASSERT(ctx != NULL);

	Graph_ResetReservedNode(ctx->gc->g);

	if(ctx->undo_log == NULL) return;
	
	UndoLog_Rollback(ctx->undo_log, ctx->gc);
	ctx->undo_log = NULL;
}

// retrieve effects-buffer
EffectsBuffer *QueryCtx_GetEffectsBuffer(void) {
	QueryCtx *ctx = _QueryCtx_GetCtx();
	ASSERT(ctx != NULL);

	if(ctx->effects_buffer == NULL) {
		ctx->effects_buffer = EffectsBuffer_New();
	}
	
	return ctx->effects_buffer;
}

// retrieve the Redis module context
RedisModuleCtx *QueryCtx_GetRedisModuleCtx(void) {
	QueryCtx *ctx = _QueryCtx_GetCtx();
	ASSERT(ctx != NULL);
	return ctx->global_exec_ctx.redis_ctx;
}

// retrieve the bolt client
bolt_client_t *QueryCtx_GetBoltClient(void) {
	QueryCtx *ctx = _QueryCtx_GetCtx();
	ASSERT(ctx != NULL);
	return ctx->global_exec_ctx.bolt_client;
}

// retrive the resultset
ResultSet *QueryCtx_GetResultSet(void) {
	QueryCtx *ctx = _QueryCtx_GetCtx();
	ASSERT(ctx != NULL);
	return ctx->internal_exec_ctx.result_set;
}

// retrive the resultset statistics
ResultSetStatistics *QueryCtx_GetResultSetStatistics(void) {
	ResultSetStatistics  *stats       =  NULL;
	ResultSet            *result_set  =  QueryCtx_GetResultSet();
	if(result_set) stats = &result_set->stats;
	return stats;
}

// print the current query
void QueryCtx_PrintQuery(void) {
	QueryCtx *ctx = _QueryCtx_GetCreateCtx();
	printf("%s\n", ctx->query_data.query);
}

static void _QueryCtx_ThreadSafeContextLock
(
	QueryCtx *ctx
) {
	// acquire GIL if we're running with a blocked client
	// implicates we're running on a worker thread and not on Redis main thread
	if(ctx->global_exec_ctx.bc) {
		RedisModule_ThreadSafeContextLock(ctx->global_exec_ctx.redis_ctx);
	} else {
		// no blocked client, most likely we're running on Redis main thread
		// ASSERT(pthread_equal(Globals_Get_MainThreadId(), pthread_self()) != 0);

		// it likely we're here because of an execution of a write query
		// e.g. loading from AOF
		// to alow Redis to reply to PING requests we should yield execution
		// giving Redis the opportunity to process commands
		// see RedisModule_Yield docs
		RedisModule_Yield(ctx->global_exec_ctx.redis_ctx,
				REDISMODULE_YIELD_FLAG_CLIENTS, NULL);
	}
}

static void _QueryCtx_ThreadSafeContextUnlock
(
	QueryCtx *ctx
) {
	if(ctx->global_exec_ctx.bc) RedisModule_ThreadSafeContextUnlock(ctx->global_exec_ctx.redis_ctx);
}

// starts a locking flow before commiting changes
// Locking flow:
// 1. lock GIL
// 2. open key with `write` flag
// 3. graph R/W lock with write flag
// since 2PL protocal is implemented, the method returns true if
// it managed to achieve locks in this call or a previous call
// in case that the locks are already locked, there will be no attempt to lock
// them again this method returns false if the key has changed
// from the current graph, and sets the relevant error message
bool QueryCtx_LockForCommit(void) {
	QueryCtx *ctx = _QueryCtx_GetCreateCtx();
	if(ctx->internal_exec_ctx.locked_for_commit) return true;

	// lock GIL
	RedisModuleCtx *redis_ctx = ctx->global_exec_ctx.redis_ctx;
	GraphContext *gc = ctx->gc;
	RedisModuleString *graphID = RedisModule_CreateString(redis_ctx, gc->graph_name,
														  strlen(gc->graph_name));
	_QueryCtx_ThreadSafeContextLock(ctx);

	// open key and verify
	RedisModuleKey *key = RedisModule_OpenKey(redis_ctx, graphID, REDISMODULE_WRITE);
	RedisModule_FreeString(redis_ctx, graphID);
	if(RedisModule_KeyType(key) == REDISMODULE_KEYTYPE_EMPTY) {
		ErrorCtx_SetError(EMSG_EMPTY_KEY, ctx->gc->graph_name);
		goto clean_up;
	}
	if(RedisModule_ModuleTypeGetType(key) != GraphContextRedisModuleType) {
		ErrorCtx_SetError(EMSG_NON_GRAPH_KEY, ctx->gc->graph_name);
		goto clean_up;

	}
	if(gc != RedisModule_ModuleTypeGetValue(key)) {
		ErrorCtx_SetError(EMSG_DIFFERENT_VALUE, ctx->gc->graph_name);
		goto clean_up;
	}
	ctx->internal_exec_ctx.key = key;

	// acquire graph write lock
	Graph_AcquireWriteLock(gc->g);
	ctx->internal_exec_ctx.locked_for_commit = true;

	return true;

clean_up:
	// free key handle
	RedisModule_CloseKey(key);

	// unlock GIL
	_QueryCtx_ThreadSafeContextUnlock(ctx);

	// if there is a break point for runtime exception, raise it, otherwise return false
	ErrorCtx_RaiseRuntimeException(NULL);
	return false;
}

static void _QueryCtx_UnlockCommit
(
	QueryCtx *ctx
) {
	GraphContext *gc = ctx->gc;

	ctx->internal_exec_ctx.locked_for_commit = false;
	// release graph R/W lock
	Graph_ReleaseLock(gc->g);

	// close Key
	RedisModule_CloseKey(ctx->internal_exec_ctx.key);

	// unlock GIL
	_QueryCtx_ThreadSafeContextUnlock(ctx);
}

// starts an ulocking flow and notifies Redis after commiting changes
// the only writer which allow to perform the unlock and commit (replicate)
// is the last_writer the method get an OpBase and compares it to
// the last writer, if they are equal then the commit and unlock flow will start
// Unlocking flow:
// 1. replicate
// 2. unlock graph R/W lock
// 3. close key
// 4. unlock GIL
void QueryCtx_UnlockCommit() {
	QueryCtx *ctx = _QueryCtx_GetCtx();
	if(!ctx) return;

	// already unlocked?
	if(!ctx->internal_exec_ctx.locked_for_commit) return;

	_QueryCtx_UnlockCommit(ctx);
}

// replicate command to AOF/Replicas
void QueryCtx_Replicate
(
	QueryCtx *ctx
) {
	ASSERT(ctx != NULL);

	GraphContext   *gc        = ctx->gc;
	RedisModuleCtx *redis_ctx = ctx->global_exec_ctx.redis_ctx;

	// replicate
	RedisModule_Replicate(redis_ctx, ctx->global_exec_ctx.command_name,
			"cc!", gc->graph_name, ctx->query_data.query);
}

// compute and return elapsed query execution time
double QueryCtx_GetRuntime(void) {
	QueryCtx *ctx = _QueryCtx_GetCtx();
	ASSERT(ctx != NULL);

	return ctx->stats.durations[QueryStage_EXECUTING] +
		ctx->stats.durations[QueryStage_REPORTING];
}

// returns query received timestamp
uint64_t QueryCtx_GetReceivedTS (void) {
	QueryCtx *ctx = _QueryCtx_GetCtx();
	ASSERT(ctx != NULL);

	return ctx->stats.received_ts ;
}

// free the allocations within the QueryCtx and reset it for the next query
void QueryCtx_Free(void) {
	QueryCtx *ctx = _QueryCtx_GetCtx();
	ASSERT(ctx != NULL);

	if(ctx->undo_log) {
		UndoLog_Free(ctx->undo_log);
		ctx->undo_log = NULL;
	}
	EffectsBuffer_Free(ctx->effects_buffer);

	if(ctx->query_data.params != NULL) {
		HashTableRelease(ctx->query_data.params);
		ctx->query_data.params = NULL;
	}

	rm_free(ctx);

	// NULL-set the context for reuse the next time this thread receives a query
	QueryCtx_RemoveFromTLS();
}

