/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "ast/ast.h"
#include "redismodule.h"
#include "util/rmalloc.h"
#include "cypher-parser.h"
#include "effects/effects.h"
#include "util/simple_timer.h"
#include "undo_log/undo_log.h"
#include "graph/graphcontext.h"
#include "resultset/resultset.h"
#include "commands/cmd_context.h"
#include "execution_plan/ops/op.h"

#include <pthread.h>

extern pthread_key_t _tlsQueryCtxKey;  // Thread local storage query context key.

// holds the execution type flags: certain traits of query regarding its
// execution
typedef enum QueryExecutionTypeFlag {
	// indicates that this query is a read-only query
	QueryExecutionTypeFlag_READ = 0,
	// indicates that this query is a write query
	QueryExecutionTypeFlag_WRITE = 1 << 0,
	// whether or not we want to profile the query
	QueryExecutionTypeFlag_PROFILE = 1 << 1,
} QueryExecutionTypeFlag;

// holds the query execution status
typedef enum QueryExecutionStatus {
	QueryExecutionStatus_SUCCESS = 0,
	QueryExecutionStatus_FAILURE,
	QueryExecutionStatus_TIMEDOUT,
} QueryExecutionStatus;

// stages a query may be in
typedef enum QueryStage {
    QueryStage_WAITING   = 0,
    QueryStage_EXECUTING = 1,
    QueryStage_REPORTING = 2,
    QueryStage_FINISHED  = 3,
} QueryStage;

typedef struct {
	AST *ast;                     // the scoped AST associated with this query
	dict *params;                 // dict [param_name, param_value]
	const char *query;            // query string
	uint query_params_len;        // length of query parameters
	const char *query_no_params;  // query string without parameters part
} QueryCtx_QueryData;

typedef struct {
	RedisModuleKey *key;     // graph open key, for later extraction and closing
	ResultSet *result_set;   // execution result set
	bool locked_for_commit;  // indicates if QueryCtx_LockForCommit been called
} QueryCtx_InternalExecCtx;

typedef struct {
	RedisModuleCtx *redis_ctx;     // the Redis module context
	RedisModuleBlockedClient *bc;  // blocked client
	bolt_client_t *bolt_client;    // bolt client
	const char *command_name;      // command name
} QueryCtx_GlobalExecCtx;

// query statistics
typedef struct {
	simple_timer_t timer;  // stage timer
	uint64_t received_ts;  // query received timestamp
	double durations[3];   // stage durations
	bool parameterized;    // uses parameters
	bool utilized_cache;   // utilized cache
} QueryStats;

typedef struct QueryCtx {
	QueryStats stats;                            // query statistics
	GraphContext *gc;                            // GraphContext associated with this query's graph
	UndoLog undo_log;                            // undo-log in case rollback is needed
	QueryStage stage;                            // query execution stage
	bool deterministic;                          // false if query contains a non deterministic element
	QueryExecutionStatus status;                 // query execution status
	QueryExecutionTypeFlag flags;                // execution flags
	EffectsBuffer *effects_buffer;               // effects-buffer for replication, used when write query succeed and replication is needed
	QueryCtx_QueryData query_data;               // data related to the query syntax
	QueryCtx_GlobalExecCtx global_exec_ctx;      // data related to global redis execution
	QueryCtx_InternalExecCtx internal_exec_ctx;  // data related to internal query execution
} QueryCtx;

// instantiate the thread-local QueryCtx on module load
bool QueryCtx_Init(void);

// retrieve this thread's QueryCtx
QueryCtx *QueryCtx_GetQueryCtx(void);

// set the provided QueryCtx in this thread's storage key
void QueryCtx_SetTLS
(
	QueryCtx *query_ctx
);

// Null-set this thread's storage key
void QueryCtx_RemoveFromTLS(void);

//------------------------------------------------------------------------------
// query statistics
//------------------------------------------------------------------------------

// advance query's stage
// waiting   -> executing
// executing -> reporting
// reporting -> finished
void QueryCtx_AdvanceStage
(
	QueryCtx *ctx  // query context
);

// reset query's stage
// waiting <- executing
void QueryCtx_ResetStage
(
	QueryCtx *ctx  // query context
);

// sets the "utilized_cache" flag of a QueryInfo
void QueryCtx_SetUtilizedCache
(
    QueryCtx *ctx,  // query context
    bool utilized   // cache utilized
);

//------------------------------------------------------------------------------
// setters
//------------------------------------------------------------------------------

// sets the global execution context
void QueryCtx_SetGlobalExecutionCtx
(
	CommandCtx *cmd_ctx
);

// set the provided AST for access through the QueryCtx
void QueryCtx_SetAST
(
	AST *ast
);

// set the provided GraphCtx for access through the QueryCtx
void QueryCtx_SetGraphCtx
(
	GraphContext *gc
);

// set the resultset
void QueryCtx_SetResultSet
(
	ResultSet *result_set
);

// set the parameters map
void QueryCtx_SetParams
(
	dict *params
);

// mark query context as not deterministic
void QueryCtx_SetNonDeterministic (void) ;

// returns true if query is deterministic
bool QueryCtx_IsDeterministic (void) ;

//------------------------------------------------------------------------------
// getters
//------------------------------------------------------------------------------

// retrieve the AST
AST *QueryCtx_GetAST(void);

// retrieve the query parameters values map
dict *QueryCtx_GetParams(void);

// retrieve the GraphCtx
GraphContext *QueryCtx_GetGraphCtx(void);

// retrieve the bolt client
bolt_client_t *QueryCtx_GetBoltClient(void);

// retrieve the Graph object
Graph *QueryCtx_GetGraph(void);

// retrieve undo log
UndoLog QueryCtx_GetUndoLog(void);

// rollback the current command
void QueryCtx_Rollback(void);

// retrieve effects-buffer
EffectsBuffer *QueryCtx_GetEffectsBuffer(void);

// retrieve the Redis module context
RedisModuleCtx *QueryCtx_GetRedisModuleCtx(void);

// retrive the resultset
ResultSet *QueryCtx_GetResultSet(void);

// retrive the resultset statistics
ResultSetStatistics *QueryCtx_GetResultSetStatistics(void);

// print the current query
void QueryCtx_PrintQuery(void);

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
bool QueryCtx_LockForCommit(void);

// starts an ulocking flow and notifies Redis after commiting changes
// the only writer which allow to perform the unlock and commit (replicate)
// is the last_writer the method get an OpBase and compares it to
// the last writer, if they are equal then the commit and unlock flow will start
// Unlocking flow:
// 1. replicate
// 2. unlock graph R/W lock
// 3. close key
// 4. unlock GIL
void QueryCtx_UnlockCommit(void);

// replicate command to AOF/Replicas
void QueryCtx_Replicate
(
	QueryCtx *ctx
);

// compute and return elapsed query execution time
double QueryCtx_GetRuntime(void);

// returns query received timestamp
uint64_t QueryCtx_GetReceivedTS (void) ;

// free the allocations within the QueryCtx and reset it for the next query
void QueryCtx_Free(void);

