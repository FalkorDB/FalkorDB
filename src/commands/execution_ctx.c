/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "execution_ctx.h"
#include "RG.h"
#include "../query_ctx.h"
#include "../errors/errors.h"
#include "../util/arr.h"
#include "../util/cache/cache.h"
#include "../util/rax_extensions.h"
#include "../arithmetic/arithmetic_expression.h"
#include "../ast/ast_shared.h"
#include "../execution_plan/ops/op_create.h"
#include "../execution_plan/ops/op_merge_create.h"
#include "../execution_plan/ops/op_unwind.h"
#include "../execution_plan/execution_plan_clone.h"
#include "../execution_plan/optimizations/optimizer.h"

static inline size_t _ArrayUsage
(
	const void *arr
) {
	return (arr != NULL) ? arr_sizeof(arr_hdr(arr)) : 0;
}

static size_t _AR_EXP_MemoryUsage
(
	const AR_ExpNode *exp
) {
	if (exp == NULL) {
		return 0;
	}

	size_t n = sizeof(*exp);
	if (AR_EXP_IsOperation(exp)) {
		n += sizeof(AR_ExpNode *) * exp->op.child_count;
		for (int i = 0; i < exp->op.child_count; i++) {
			n += _AR_EXP_MemoryUsage(exp->op.children[i]);
		}
	} else if (AR_EXP_IsConstant(exp)) {
		n += SIValue_memoryUsage(exp->operand.constant);
	}

	return n;
}

static size_t _PropertyMap_MemoryUsage
(
	const PropertyMap *map
) {
	if (map == NULL) {
		return 0;
	}

	size_t n = sizeof(*map);
	n += _ArrayUsage(map->keys);
	n += _ArrayUsage(map->attr_ids);
	n += _ArrayUsage(map->values);

	uint val_count = arr_len(map->values);
	for (uint i = 0; i < val_count; i++) {
		n += _AR_EXP_MemoryUsage(map->values[i]);
	}

	return n;
}

static size_t _NodeCreateCtx_MemoryUsage
(
	const NodeCreateCtx *ctx
) {
	if (ctx == NULL) {
		return 0;
	}

	size_t n = sizeof(*ctx);
	n += _ArrayUsage(ctx->labels);
	n += _ArrayUsage(ctx->labelsId);
	n += _PropertyMap_MemoryUsage(ctx->properties);
	return n;
}

static size_t _EdgeCreateCtx_MemoryUsage
(
	const EdgeCreateCtx *ctx
) {
	if (ctx == NULL) {
		return 0;
	}

	size_t n = sizeof(*ctx);
	n += _PropertyMap_MemoryUsage(ctx->properties);
	return n;
}

static size_t _PendingCreationsTemplateMemoryUsage
(
	const PendingCreations *pending
) {
	if (pending == NULL) {
		return 0;
	}

	size_t n = 0;
	n += _ArrayUsage(pending->nodes.nodes_to_create);
	n += _ArrayUsage(pending->nodes.node_attributes);
	n += _ArrayUsage(pending->nodes.node_labels);
	n += _ArrayUsage(pending->nodes.created_nodes);

	uint node_count = arr_len(pending->nodes.nodes_to_create);
	for (uint i = 0; i < node_count; i++) {
		n += _NodeCreateCtx_MemoryUsage(&pending->nodes.nodes_to_create[i]);
	}

	n += _ArrayUsage(pending->edges);
	uint edge_count = arr_len(pending->edges);
	for (uint i = 0; i < edge_count; i++) {
		const PendingEdgeCreations *edge = &pending->edges[i];
		n += sizeof(*edge);
		n += _EdgeCreateCtx_MemoryUsage(&edge->edges_to_create);
		n += _ArrayUsage(edge->edge_attributes);
		n += _ArrayUsage(edge->created_edges);
	}

	return n;
}

static size_t _OpSpecificTemplateMemoryUsage
(
	const OpBase *op
) {
	switch (op->type) {
	case OPType_CREATE:
		return _PendingCreationsTemplateMemoryUsage(&((const OpCreate *)op)->pending);
	case OPType_MERGE_CREATE:
		return _PendingCreationsTemplateMemoryUsage(&((const OpMergeCreate *)op)->pending);
	case OPType_UNWIND:
		return _AR_EXP_MemoryUsage(((const OpUnwind *)op)->exp);
	default:
		return 0;
	}
}

static size_t _AST_MemoryUsage
(
	const AST *ast
) {
	if (ast == NULL) {
		return 0;
	}

	size_t n = sizeof(*ast);
	n += sizeof(*ast->ref_count);
	n += sizeof(*ast->anot_ctx_collection);
	n += raxMemoryUsage(ast->referenced_entities);

	return n;
}

static size_t _ExecutionPlan_MemoryUsage
(
	const ExecutionPlan *plan
) {
	if (plan == NULL) {
		return 0;
	}

	size_t n = sizeof(*plan);
	n += raxMemoryUsage(plan->record_map);
	n += (plan->query_graph != NULL) ? sizeof(*plan->query_graph) : 0;
	n += (plan->record_pool != NULL) ? sizeof(*plan->record_pool) : 0;

	if (plan->root == NULL) {
		return n;
	}

	// follow the same traversal shape as ExecutionPlan_Free.
	OpBase **to_visit = arr_new(OpBase *, 1);
	arr_append(to_visit, plan->root);

	while (arr_len(to_visit) > 0) {
		OpBase *op = arr_pop(to_visit);
		n += sizeof(*op);
		n += (op->childCount > 0) ? (sizeof(OpBase *) * op->childCount) : 0;
		n += (op->stats != NULL) ? sizeof(*op->stats) : 0;
		n += _OpSpecificTemplateMemoryUsage(op);

		for (uint i = 0; i < op->childCount; i++) {
			if (op->children[i] != NULL) {
				arr_append(to_visit, op->children[i]);
			}
		}
	}

	arr_free(to_visit);
	return n;
}

static inline void _LogQueryCacheMemoryUsage
(
	Cache *cache,
	const char *phase,
	const char *result
) {
	size_t usage = Cache_MemoryUsage(cache, (CacheEntryMemUsageFunc)ExecutionCtx_MemoryUsage);
	printf("[query-cache] phase=%s result=%s entries=%u/%u memory=%zu bytes\n",
		phase,
		result,
		cache->size,
		cache->cap,
		usage);
}

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

	// if parameter parsing set an error, bail out before the (potentially
	// corrupted) query buffer reaches the Cypher parser
	if (unlikely (ErrorCtx_EncounteredError ())) {
		return NULL ;
	}

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
	_LogQueryCacheMemoryUsage(cache, "lookup", ret != NULL ? "hit" : "miss");

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
		_LogQueryCacheMemoryUsage(cache, "store",
				(ret != exec_ctx) ? "inserted" : "existing");
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

size_t ExecutionCtx_MemoryUsage
(
	const ExecutionCtx *ctx
) {
	if(ctx == NULL) {
		return 0;
	}

	size_t n = sizeof(*ctx);
	n += _AST_MemoryUsage(ctx->ast);
	n += _ExecutionPlan_MemoryUsage(ctx->plan);

	return n;
}
