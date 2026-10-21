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
#include "configuration/config.h"
#include "arithmetic/arithmetic_expression.h"
#include "serializers/graphcontext_type.h"

#include <errno.h>
#include <time.h>

// GraphContext type as it is registered at Redis
extern RedisModuleType *GraphContextRedisModuleType;

pthread_key_t _tlsQueryCtxKey;  // thread local storage query context key

// retrieve or instantiate new QueryCtx
static inline QueryCtx *_QueryCtx_GetCreateCtx(void) {
	QueryCtx *ctx = pthread_getspecific(_tlsQueryCtxKey);

	if(ctx == NULL) {
		// set a new thread-local QueryCtx if one has not been created
		ctx = rm_calloc (1, sizeof(QueryCtx)) ;

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

// returns the number of milliseconds elapsed for the current stage
double QueryCtx_StageElapsed
(
	const QueryCtx *ctx  // query context
) {
	ASSERT (ctx != NULL) ;

	return TIMER_GET_ELAPSED_MILLISECONDS (ctx->stats.timer) ;
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
Graph *QueryCtx_GetGraph (void) {
	GraphContext *gc = QueryCtx_GetGraphCtx() ;
	return GraphContext_GetGraph (gc) ;
}

// retrieve undo log
UndoLog QueryCtx_GetUndoLog (void) {
	QueryCtx *ctx = _QueryCtx_GetCtx () ;
	ASSERT (ctx != NULL) ;
	
	if (ctx->undo_log == NULL) {
		ctx->undo_log = UndoLog_New () ;
	}

	return ctx->undo_log ;
}

// rollback the current command
void QueryCtx_Rollback (void) {
	QueryCtx *ctx = _QueryCtx_GetCtx () ;
	ASSERT (ctx != NULL) ;

	Graph_ResetReservedNode (GraphContext_GetGraph (ctx->gc)) ;

	if (ctx->undo_log == NULL) {
		return ;
	}
	
	UndoLog_Rollback (&ctx->undo_log) ;
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

// retrive the resultset
ResultSet *QueryCtx_GetResultSet(void) {
	QueryCtx *ctx = _QueryCtx_GetCtx();
	ASSERT(ctx != NULL);
	return ctx->internal_exec_ctx.result_set;
}

// retrive the resultset statistics
ResultSetStatistics *QueryCtx_GetResultSetStatistics (void) {
	ResultSetStatistics  *stats      = NULL ;
	ResultSet            *result_set = QueryCtx_GetResultSet () ;

	if (result_set) {
		stats = &result_set->stats ;
	}

	return stats ;
}

// retrieve the query execution type
QueryExecutionTypeFlag QueryCtx_GetExecutionType (void) {
	QueryCtx *ctx = _QueryCtx_GetCtx () ;
	return (ctx) ? ctx->flags : QueryExecutionTypeFlag_READ ;
}

// print the current query
void QueryCtx_PrintQuery(void) {
	QueryCtx *ctx = _QueryCtx_GetCreateCtx();
	printf("%s\n", ctx->query_data.query);
}

//------------------------------------------------------------------------------
// Lock management
//------------------------------------------------------------------------------

static void _QueryCtx_ThreadSafeContextLock
(
	QueryCtx *ctx
) {
	// acquire GIL if we're running with a blocked client
	// implicates we're running on a worker thread and not on Redis main thread
	if (ctx->global_exec_ctx.bc) {
		RedisModule_ThreadSafeContextLock (ctx->global_exec_ctx.redis_ctx) ;
	} else {
		// no blocked client, most likely we're running on Redis main thread
		// ASSERT(pthread_equal(Globals_Get_MainThreadId(), pthread_self()) != 0);

		// yield only during AOF/RDB loading so that Redis can respond to
		// health-check PINGs (e.g. from Sentinel or external monitors)
		// while replaying potentially long-running write commands.
		//
		// we must NOT yield during replication or MULTI/EXEC because
		// RedisModule_Yield with REDISMODULE_YIELD_FLAG_CLIENTS puts
		// Redis into a "busy" state, causing other clients to receive
		// BUSY errors (JedisBusyException)
		int flags = RedisModule_GetContextFlags (ctx->global_exec_ctx.redis_ctx) ;
		if (flags & REDISMODULE_CTX_FLAGS_LOADING) {
			RedisModule_Yield (ctx->global_exec_ctx.redis_ctx,
					REDISMODULE_YIELD_FLAG_CLIENTS, NULL) ;
		}
	}
}

static void _QueryCtx_ThreadSafeContextUnlock
(
	QueryCtx *ctx
) {
	if (ctx->global_exec_ctx.bc != NULL) {
		RedisModule_ThreadSafeContextUnlock (ctx->global_exec_ctx.redis_ctx) ;
	}
}

// acquire graph's read lock
void QueryCtx_AcquireReadLock (void) {
	QueryCtx *ctx = _QueryCtx_GetCreateCtx () ;

	// shouldn't try to acquire write lock when read lock is held
	ASSERT (ctx->internal_exec_ctx.write_locked == false) ;

	// return if we've already acquired read lock
	if (ctx->internal_exec_ctx.read_locked == true) {
		return ;
	}

	// acquire read lock
	GraphContext_AcquireReadLock (QueryCtx_GetGraphCtx ()) ;
	ctx->internal_exec_ctx.read_locked = true ;
}

// acquire graph's read lock for a reader query, first waiting out any writer
// on this graph that is acquiring the write lock (the writer gate) and bounding
// the total wait by timeout_ms (<= 0 blocks)
// returns true if the read lock was acquired, false if the timeout elapsed
// first (caller should abort the query as a timeout)
bool QueryCtx_TimeAcquireReadLock
(
	int64_t timeout_ms
) {
	QueryCtx *ctx = _QueryCtx_GetCreateCtx () ;

	// shouldn't try to acquire read lock when write lock is held
	ASSERT (ctx->internal_exec_ctx.write_locked == false) ;

	// return if we've already acquired read lock
	if (ctx->internal_exec_ctx.read_locked == true) {
		return true ;
	}

	int rc = GraphContext_TimeAcquireReadLock (QueryCtx_GetGraphCtx (),
			timeout_ms) ;
	if (rc == 0) {
		ctx->internal_exec_ctx.read_locked = true ;
		return true ;
	}

	return false ;
}

// outcome of a single write-lock acquisition attempt
typedef enum {
	WRITE_LOCK_ACQUIRED,  // lock held; GIL and key handle retained for caller
	WRITE_LOCK_RETRY,     // poll interval elapsed; GIL released, caller retries
	WRITE_LOCK_FAILED     // validation / lock error; GIL released, caller aborts
} WriteLockAttempt;

// perform a single attempt to acquire the graph write lock:
// lock the GIL, re-validate pause / role / key, then wait up to `poll_interval`
// milliseconds for the write lock
//
// the full re-validation runs on every attempt because the GIL is released
// between attempts: the key may have been dropped / replaced, this instance
// may have been demoted to a replica, or a pause window may have opened in
// the meantime
//
// on WRITE_LOCK_ACQUIRED the GIL and the open key handle are retained for the
// remainder of the write; on every other outcome both are released
static WriteLockAttempt _QueryCtx_TryAcquireWriteLock (
	QueryCtx *ctx,           // query context
	GraphContext *gc,        // graph to lock
	int64_t poll_interval    // ms to wait for the lock before yielding the GIL
) {
	RedisModuleCtx *redis_ctx = ctx->global_exec_ctx.redis_ctx ;
	WriteLockAttempt result   = WRITE_LOCK_FAILED ;
	int rc                    = 0 ;

	//--------------------------------------------------------------------------
	// lock GIL
	//--------------------------------------------------------------------------

	_QueryCtx_ThreadSafeContextLock (ctx) ;

	// read the graph name only after taking the GIL. between attempts the GIL
	// is released, and a RENAME (which runs on the main thread under the GIL)
	// can free and replace gc->graph_name; capturing it before the GIL - or
	// reusing a pointer captured once by the caller - would dereference freed
	// memory on a later attempt. holding the GIL pins the name for this attempt
	const char *graph_name = GraphContext_GetName (gc) ;
	RedisModuleString *graphID = RedisModule_CreateString (redis_ctx,
			graph_name, strlen (graph_name)) ;

	RedisModuleKey *key = RedisModule_OpenKey (redis_ctx, graphID,
			REDISMODULE_WRITE) ;
	RedisModule_FreeString (redis_ctx, graphID) ;

	//--------------------------------------------------------------------------
	// re-validate pause / role now that the GIL is held
	//--------------------------------------------------------------------------
	// a write admitted before a replica-pause window opened (e.g. FAILOVER /
	// CLIENT PAUSE) may reach this point after the pause started, or after
	// this instance has been demoted to a replica. once the GIL is held,
	// neither can change until we release it, so this check is race-free.
	//
	// commands applied from our own master's replication stream, or being
	// replayed while loading AOF/RDB, are exempt: they were already
	// committed elsewhere (or persisted locally) - our job is to mirror
	// them, not re-authorize them

	int ctx_flags = RedisModule_GetContextFlags (redis_ctx) ;
	bool bypass_checks = (ctx_flags & REDISMODULE_CTX_FLAGS_REPLICATED)    ||
	                      (ctx_flags & REDISMODULE_CTX_FLAGS_LOADING)       ||
	                      (ctx_flags & REDISMODULE_CTX_FLAGS_ASYNC_LOADING) ;

	if (!bypass_checks) {
		if (RedisModule_AvoidReplicaTraffic ()) {
			ErrorCtx_SetError (EMSG_REPLICA_TRAFFIC_PAUSED) ;
			goto done ;
		}

		if (ctx_flags & REDISMODULE_CTX_FLAGS_READONLY) {
			ErrorCtx_SetError (EMSG_NOT_MASTER) ;
			goto done ;
		}
	}

	if (RedisModule_KeyType (key) == REDISMODULE_KEYTYPE_EMPTY) {
		ErrorCtx_SetError (EMSG_EMPTY_KEY, graph_name) ;
		goto done ;
	}

	if (RedisModule_ModuleTypeGetType (key) != GraphContextRedisModuleType) {
		ErrorCtx_SetError (EMSG_NON_GRAPH_KEY, graph_name) ;
		goto done ;
	}

	if (gc != RedisModule_ModuleTypeGetValue (key)) {
		ErrorCtx_SetError (EMSG_DIFFERENT_VALUE, graph_name) ;
		goto done ;
	}

	//--------------------------------------------------------------------------
	// try to acquire the graph write lock within the poll interval
	//--------------------------------------------------------------------------

	rc = GraphContext_TimeAcquireWriteLock (gc, poll_interval) ;
	if (rc == 0) {
		// success: retain GIL and key handle for the remainder of the write
		ctx->internal_exec_ctx.key = key ;
		ctx->internal_exec_ctx.write_locked = true ;
		ASSERT (ctx->internal_exec_ctx.read_locked == false) ;
		return WRITE_LOCK_ACQUIRED ;
	}

	// ETIMEDOUT: poll interval elapsed, retry after yielding the GIL
	// any other return is unexpected and aborts the write
	result = (rc == ETIMEDOUT) ? WRITE_LOCK_RETRY : WRITE_LOCK_FAILED ;
	if (result == WRITE_LOCK_FAILED) {
		ErrorCtx_SetError (EMSG_WRITE_LOCK_FAILED, rc) ;
	}

done:
	// release key handle and GIL before retrying / aborting
	RedisModule_CloseKey (key) ;
	_QueryCtx_ThreadSafeContextUnlock (ctx) ;
	return result ;
}

// acquire graph's write lock
bool QueryCtx_AcquireWriteLock (void) {
	QueryCtx *ctx = _QueryCtx_GetCreateCtx () ;
	GraphContext *gc = ctx->gc ;

	// return if we've already acquired write lock
	if (ctx->internal_exec_ctx.write_locked == true) {
		ASSERT (ctx->internal_exec_ctx.read_locked == false) ;
		return true ;
	}

	// locking flow:
	// 1. release read lock
	// 2. lock GIL
	// 3. open key with `write` flag
	// 4. graph R/W lock with write flag
	// since 2PL protocal is implemented, the method returns true if
	// it managed to achieve locks in this call or a previous call
	// in case that the locks are already locked, there will be no attempt to lock
	// them again this method returns false if the key has changed
	// from the current graph, and sets the relevant error message

	// release graph READ lock if held
	// writer queries hold a READ lock during the match phase to prevent
	// concurrent memory modifications by defrag
	// must release before acquiring GIL to avoid deadlock:
	//   worker: READ lock → GIL  vs  main thread: GIL → WRITE lock
	bool read_locked = ctx->internal_exec_ctx.read_locked ;
	if (read_locked == true) {
		GraphContext_ReleaseReadLock (gc) ;
		ctx->internal_exec_ctx.read_locked = false ;
	}

	// initial slice: ms to park on the graph lock per attempt before yielding
	int64_t poll_interval ;
	Config_Option_get (Config_WRITE_LOCK_POLL_INTERVAL, &poll_interval) ;

	//--------------------------------------------------------------------------
	// acquire the write lock, yielding the GIL between attempts
	//--------------------------------------------------------------------------
	// holding the GIL while blocking on the write lock freezes the whole server
	// until every in-flight reader drains. instead we wait for the lock in fixed
	// `poll_interval` ms slices: each attempt parks on the lock for at most that
	// long (holding the GIL), then releases the GIL and sleeps briefly before
	// re-validating and retrying - so the main thread stays responsive for the
	// whole wait, however long the in-flight reads run.
	//
	// the short sleep is the actual hand-off: without it the worker re-acquires
	// the GIL before the blocked main thread can grab it, so the GIL would stay
	// effectively pinned. sleeping off-CPU guarantees the main thread wins the
	// GIL and serves pending commands between slices.
	//
	// anti-starvation is the writer gate's job, not this loop's: new reader-
	// queries park at the gate (below) instead of taking the read lock, so they
	// can't slip in during our hand-offs. the writer therefore waits only for
	// the readers already holding the lock to drain - bounded, and responsive
	// the whole time. no backoff, no blocking escalation: both only traded that
	// responsiveness away for a landing guarantee the gate already provides.
	GraphContext_WriterGateEnter (gc) ;

	WriteLockAttempt attempt ;

	if (poll_interval <= 0) {
		// legacy mode: no slicing / yielding. acquire the lock with a single
		// blocking call that holds the GIL until granted - the exact behavior
		// prior to the responsive write-lock change. the server may stall here
		// until all in-flight readers drain
		attempt = _QueryCtx_TryAcquireWriteLock (ctx, gc, -1) ;
	} else {
		const struct timespec yield = { 0, 1000000 } ;  // 1ms GIL hand-off
		while ((attempt = _QueryCtx_TryAcquireWriteLock (ctx, gc,
						poll_interval)) == WRITE_LOCK_RETRY) {
			// GIL released by the attempt; sleep off-CPU so the main thread runs
			nanosleep (&yield, NULL) ;
		}
	}

	// writer no longer waiting: wake readers parked at the gate. on success
	// they proceed and block on the now-held rwlock; on failure they proceed
	// normally
	GraphContext_WriterGateLeave (gc) ;

	if (attempt == WRITE_LOCK_ACQUIRED) {
		return true ;
	}

	// acquisition failed: restore the read lock if we released one, then raise
	// the runtime exception breakpoint if set, otherwise return false
	if (read_locked) {
		QueryCtx_AcquireReadLock () ;
	}

	ErrorCtx_RaiseRuntimeException (NULL) ;
	return false ;
}

// release graph's lock
void QueryCtx_ReleaseLock (void) {
	QueryCtx *ctx = _QueryCtx_GetCtx () ;
	ASSERT (ctx != NULL) ;

	// return if neither lock is held
	if (ctx->internal_exec_ctx.write_locked == false &&
		ctx->internal_exec_ctx.read_locked  == false) {
		return ;
	}

	// starts an ulocking flow
	// Unlocking flow:
	// 1. unlock graph R/W lock
	// 2. close key
	// 3. unlock GIL

	// release graph lock
	GraphContext_ReleaseLock (QueryCtx_GetGraphCtx ()) ;

	// release WRITE lock
	if (ctx->internal_exec_ctx.write_locked == true) {
		// close Key
		RedisModule_CloseKey (ctx->internal_exec_ctx.key) ;

		// unlock GIL
		_QueryCtx_ThreadSafeContextUnlock (ctx) ;
	}
	
	ctx->internal_exec_ctx.read_locked  = false ;
	ctx->internal_exec_ctx.write_locked = false ;
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
	RedisModule_Replicate (redis_ctx, ctx->global_exec_ctx.command_name,
			"cc!", GraphContext_GetName (gc), ctx->query_data.query) ;
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
void QueryCtx_Free (void) {
	QueryCtx *ctx = _QueryCtx_GetCtx () ;

	ASSERT (ctx != NULL) ;
	ASSERT (ctx->internal_exec_ctx.read_locked  == false) ;
	ASSERT (ctx->internal_exec_ctx.write_locked == false) ;

	if (ctx->undo_log != NULL) {
		UndoLog_Free (&ctx->undo_log) ;
	}

	if (ctx->effects_buffer != NULL) {
		EffectsBuffer_Free (ctx->effects_buffer) ;
	}

	if (ctx->query_data.params != NULL) {
		HashTableRelease (ctx->query_data.params) ;
		ctx->query_data.params = NULL ;
	}

	rm_free (ctx) ;

	// NULL-set the context for reuse the next time this thread receives a query
	QueryCtx_RemoveFromTLS () ;
}

