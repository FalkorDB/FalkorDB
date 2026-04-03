/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../ast/ast.h"
#include "cmd_context.h"
#include "../execution_plan/execution_plan.h"

 // execution type derived from a query
typedef enum {
	EXECUTION_TYPE_QUERY,         // normal query execution
	EXECUTION_TYPE_INDEX_CREATE,  // create index execution
	EXECUTION_TYPE_INDEX_DROP     // drop index execution
} ExecutionType;

 // a struct for saving execution objects in cache
typedef struct {
	AST *ast;                 // AST
	bool cached;              // cache hit/miss
	bool deterministic;       // false if query contains a non deterministic exp
	ExecutionPlan *plan;      // execution plan
	ExecutionType exec_type;  // execution type: query, index create/delete
} ExecutionCtx;

// returns the objects and information required for query execution
// if the query contains error, a ExecutionCtx struct with the AST
// and Execution plan objects will be NULL
// and EXECUTION_TYPE_INVALID is returned
// returns ExecutionCtx populated with the current execution relevant objects
ExecutionCtx *ExecutionCtx_FromQuery
(
	CommandCtx *cmd_ctx
);

// clone the execution ctx and return a shallow copy for the ast
// deep copy for the execution plan
ExecutionCtx *ExecutionCtx_Clone
(
	const ExecutionCtx *ctx  // execution context to clone
);

// free an ExecutionCTX struct and its inner fields
void ExecutionCtx_Free
(
	ExecutionCtx *ctx  // execution context to free
);

