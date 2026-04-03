/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "execution_ctx.h"
#include "RG.h"
#include "../query_ctx.h"
#include "../errors/errors.h"
#include "../execution_plan/execution_plan_clone.h"
#include "../execution_plan/optimizations/optimizer.h"

static ExecutionType _GetExecutionTypeFromAST
(
	const AST *ast
) {
	const cypher_astnode_type_t root_type = cypher_astnode_type(ast->root);

	if(root_type == CYPHER_AST_QUERY) {
		return EXECUTION_TYPE_QUERY;
	}

	if(root_type == CYPHER_AST_CREATE_NODE_PROPS_INDEX) {
		return EXECUTION_TYPE_INDEX_CREATE;
	}

	if(root_type == CYPHER_AST_CREATE_PATTERN_PROPS_INDEX) {
		return EXECUTION_TYPE_INDEX_CREATE;
	}

	if(root_type == CYPHER_AST_DROP_PROPS_INDEX ||
	   root_type == CYPHER_AST_DROP_PATTERN_PROPS_INDEX) {
		return EXECUTION_TYPE_INDEX_DROP;
	}

	ASSERT(false && "Unknown execution type");
	return 0;
}

static AST *_ExecutionCtx_ParseAST
(
	const char *q_str
) {
	cypher_parse_result_t *query_parse_result = parse_query(q_str);
	// if no output from the parser, the query is not valid
	if(ErrorCtx_EncounteredError() || query_parse_result == NULL) {
		parse_result_free(query_parse_result);
		return NULL;
	}

	// prepare the constructed AST
	AST *ast = AST_Build(query_parse_result);

	return ast;
}

static ExecutionCtx *_ExecutionCtx_New
(
	AST *ast,
	ExecutionPlan *plan,
	ExecutionType exec_type,
	bool deterministic
) {
	ExecutionCtx *exec_ctx = rm_malloc (sizeof (ExecutionCtx)) ;

	exec_ctx->ast           = ast ;
	exec_ctx->plan          = plan ;
	exec_ctx->cached        = false ;
	exec_ctx->exec_type     = exec_type ;
	exec_ctx->deterministic = deterministic ;

	return exec_ctx ;
}

// clone the execution ctx and return a shallow copy for the ast
// deep copy for the execution plan
ExecutionCtx *ExecutionCtx_Clone
(
	const ExecutionCtx *ctx  // execution context to clone
) {
	ExecutionCtx *clone = rm_malloc (sizeof (ExecutionCtx)) ;

	clone->ast = AST_ShallowCopy (ctx->ast) ;

	// set the AST copy in thread local storage
	QueryCtx_SetAST (clone->ast) ;

	clone->plan          = ExecutionPlan_Clone (ctx->plan) ;
	clone->cached        = ctx->cached ;
	clone->exec_type     = ctx->exec_type ;
	clone->deterministic = ctx->deterministic ;

	return clone ;
}

// returns the objects and information required for query execution
// if the query contains error, a ExecutionCtx struct with the AST
// and Execution plan objects will be NULL
// and EXECUTION_TYPE_INVALID is returned
// returns ExecutionCtx populated with the current execution relevant objects
ExecutionCtx *ExecutionCtx_FromQuery
(
	CommandCtx *cmd_ctx
) {
	ASSERT (cmd_ctx != NULL) ;

	ExecutionCtx *ret ;

	if (unlikely (cmd_ctx->query_len == 0)) {
		ErrorCtx_SetError (EMSG_EMPTY_QUERY) ;
		return NULL ;
	}

	const char *query_no_params = cmd_ctx->query ;

	// parse and validate parameters only
	// copy the query
	ASSERT (cmd_ctx->params == NULL) ;
	cmd_ctx->params = rm_malloc (cmd_ctx->query_len + 1) ;
	memcpy (cmd_ctx->params, cmd_ctx->query, cmd_ctx->query_len) ;
	cmd_ctx->params[cmd_ctx->query_len] = '\0';

	// cmd_ctx->query string excluding query parameters
	parse_params (cmd_ctx->params, &query_no_params) ;

	// query included only params e.g. 'cypher a=1' was provided
	if (unlikely (*query_no_params == '\0')) {
		ErrorCtx_SetError (EMSG_EMPTY_QUERY) ;
		return NULL ;
	}

	// update query context with the query
	// (here the QueryInfo is created as well, starting the stage timer)
	QueryCtx *ctx = QueryCtx_GetQueryCtx () ;
	ctx->query_data.query_no_params  = query_no_params ;
	ctx->query_data.query_params_len = query_no_params - cmd_ctx->params ;

	// get cache
	Cache *cache = GraphContext_GetCache(QueryCtx_GetGraphCtx());

	// see if we already have a cached execution-ctx for given query
	ret = Cache_GetValue (cache, query_no_params) ;

	//--------------------------------------------------------------------------
	// cache hit
	//--------------------------------------------------------------------------

	if (ret != NULL) {
		ret->cached = true ;  // mark cached execution
		return ret ;
	}

	//--------------------------------------------------------------------------
	// cache miss
	//--------------------------------------------------------------------------

	// try to parse the query
	AST *ast = _ExecutionCtx_ParseAST (query_no_params) ;

	// parser failed
	if (ast == NULL) {
		// if no error has been set, emit one now
		if (!ErrorCtx_EncounteredError()) {
			ErrorCtx_SetError (EMSG_COULD_NOT_PARSE_QUERY) ;
		}
		return NULL ;
	}

	ExecutionType exec_type = _GetExecutionTypeFromAST (ast) ;

	// in case of valid query
	// create execution plan, and cache it and the AST
	if (exec_type == EXECUTION_TYPE_QUERY) {
		//----------------------------------------------------------------------
		// build execution-plan
		//----------------------------------------------------------------------
		ExecutionPlan *plan = ExecutionPlan_FromTLS_AST () ;

		// TODO: there must be a better way to understand if the execution-plan
		// was constructed correctly,
		// maybe free the plan within ExecutionPlan_FromTLS_AST, if error was
		// encountered and return NULL ?
		if (ErrorCtx_EncounteredError ()) {
			// failed to construct plan
			// clean up and return NULL
			AST_Free (ast) ;

			if (plan != NULL) {
				ExecutionPlan_Free (plan) ;
			}

			return NULL ;
		}

		// apply compile time optimizations
		Optimizer_CompileTimeOptimize (plan) ;

		// remember if query is deterministic
		ExecutionCtx *exec_ctx = _ExecutionCtx_New (ast, plan, exec_type,
				QueryCtx_IsDeterministic ()) ;
		ret = Cache_SetGetValue (cache, query_no_params, exec_ctx) ;
	} else {
		ret = _ExecutionCtx_New (ast, NULL, exec_type, true) ;
	}

	return ret ;
}

// free an ExecutionCTX struct and its inner fields
void ExecutionCtx_Free
(
	ExecutionCtx *ctx  // execution context to free
) {
	if(ctx == NULL) {
		return;
	}

	if(ctx->plan != NULL) {
		ExecutionPlan_Free(ctx->plan);
	}

	if(ctx->ast != NULL) {
		AST_Free(ctx->ast);
	}

	rm_free(ctx);
}

