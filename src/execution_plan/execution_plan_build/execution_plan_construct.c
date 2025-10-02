/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "../ops/ops.h"
#include "../../query_ctx.h"
#include "../../ast/ast_mock.h"
#include "build_call_subquery.h"
#include "execution_plan_util.h"
#include "execution_plan_modify.h"
#include "execution_plan_construct.h"
#include "../../util/rax_extensions.h"
#include "../../ast/ast_build_op_contexts.h"
#include "../../arithmetic/arithmetic_expression_construct.h"

static inline void _buildCreateOp
(
	GraphContext *gc,
	AST *ast,
	ExecutionPlan *plan,
	const cypher_astnode_t *clause
) {
	AST_CreateContext create_ast_ctx =
		AST_PrepareCreateOp(plan->query_graph, plan->record_map, clause);

	OpBase *op =
		NewCreateOp(plan, create_ast_ctx.nodes_to_create,
				create_ast_ctx.edges_to_create,
				create_ast_ctx.named_paths_aliases,
				create_ast_ctx.named_paths_elements);

	ExecutionPlan_UpdateRoot(plan, op);
}

static inline void _buildUnwindOp
(
	ExecutionPlan *plan,
	const cypher_astnode_t *clause
) {
	AST_UnwindContext unwind_ast_ctx = AST_PrepareUnwindOp(clause);
	OpBase *op = NewUnwindOp(plan, unwind_ast_ctx.exp);
	ExecutionPlan_UpdateRoot(plan, op);
}

static void _buildLoadCSVOp
(
	ExecutionPlan *plan,
	const cypher_astnode_t *clause
) {
	ASSERT(plan   != NULL);
	ASSERT(clause != NULL);

	const cypher_astnode_t *node;

	// extract information from AST

	// with headers
	bool with_headers = cypher_ast_load_csv_has_with_headers(clause);

	// alias
	node = cypher_ast_load_csv_get_identifier(clause);
	const char *alias = cypher_ast_identifier_get_name(node);

	// delimiter
	char delimiter = ',';
	node = cypher_ast_load_csv_get_field_terminator(clause);
	if(node != NULL) {
		ASSERT(cypher_astnode_type(node) == CYPHER_AST_STRING);
		const char *str_delimiter = cypher_ast_string_get_value(node) ;

		// error if delimiter is not a single character
		if (strlen (str_delimiter) != 1) {
			ErrorCtx_SetError (
					"CSV field terminator can only be one character wide") ;
			return ;
		}

		delimiter = str_delimiter[0] ;
	}

	// URI expression
	node = cypher_ast_load_csv_get_url(clause);
	AR_ExpNode *exp = AR_EXP_FromASTNode(node);

	OpBase *op = NewLoadCSVOp (plan, exp, alias, with_headers, delimiter) ;
	ExecutionPlan_UpdateRoot (plan, op) ;
}

static inline void _buildUpdateOp
(
	GraphContext *gc,
	ExecutionPlan *plan,
	const cypher_astnode_t *clause
) {
	rax *update_exps = AST_PrepareUpdateOp(gc, clause);
	OpBase *op = NewUpdateOp(plan, update_exps);
	ExecutionPlan_UpdateRoot(plan, op);
}

static inline void _buildDeleteOp
(
	ExecutionPlan *plan,
	const cypher_astnode_t *clause
) {
	if(plan->root == NULL) {
		// delete must operate on child data, prepare an error if there
		// is no child op
		ErrorCtx_SetError(EMSG_DELETE_OPERATE_ON_CHILD);
	}
	AR_ExpNode **exps = AST_PrepareDeleteOp(clause);
	OpBase *op = NewDeleteOp(plan, exps);
	ExecutionPlan_UpdateRoot(plan, op);
}

static void _buildForeachOp
(
	ExecutionPlan *plan,             // execution plan to add operation to
	const cypher_astnode_t *clause,  // foreach clause
	GraphContext *gc                 // graph context
) {
	// construct the following sub execution plan structure
	// foreach
	//   loop body (foreach/create/update/remove/delete/merge)
	//		unwind
	//			argument list

	//--------------------------------------------------------------------------
	// create embedded execution plan for the body of the Foreach clause
	//--------------------------------------------------------------------------

	// construct AST from Foreach body
	uint nclauses = cypher_ast_foreach_nclauses(clause);
	cypher_astnode_t **clauses = array_new(cypher_astnode_t *, nclauses);
	for(uint i = 0; i < nclauses; i++) {
		cypher_astnode_t *inner_clause =
			(cypher_astnode_t *)cypher_ast_foreach_get_clause(clause, i);
		array_append(clauses, inner_clause);
	}

	struct cypher_input_range range = {0};
	cypher_astnode_t *new_root = cypher_ast_query(
		NULL, 0, clauses, nclauses, clauses, nclauses, range
	);

	uint *ref_count = rm_malloc(sizeof(uint));
	*ref_count = 1;

	AST *ast = rm_malloc(sizeof(AST));

	ast->root                = new_root;
	ast->free_root           = true;
	ast->parse_result        = NULL;
	ast->ref_count           = ref_count;
	ast->anot_ctx_collection = plan->ast_segment->anot_ctx_collection;
	ast->referenced_entities = raxClone(plan->ast_segment->referenced_entities);

	// create the Foreach op, and update (outer) plan root
	OpBase *foreach = NewForeachOp(plan);
	ExecutionPlan_UpdateRoot(plan, foreach);

	ExecutionPlan *embedded_plan = ExecutionPlan_NewEmptyExecutionPlan();
	embedded_plan->ast_segment   = ast;
	embedded_plan->record_map    = raxClone(plan->record_map);

	const char **arguments = NULL;

	if(plan->root) {
		rax *bound_vars = raxNew();
		ExecutionPlan_BoundVariables(foreach, bound_vars, plan);
		arguments = (const char **)raxValues(bound_vars);
		raxFree(bound_vars);
	}

	//--------------------------------------------------------------------------
	// build Unwind op
	//--------------------------------------------------------------------------

	// unwind foreach list expression
	AR_ExpNode *exp = AR_EXP_FromASTNode(
		cypher_ast_foreach_get_expression(clause));
	exp->resolved_name = cypher_ast_identifier_get_name(
		cypher_ast_foreach_get_identifier(clause));
	OpBase *unwind = NewUnwindOp(embedded_plan, exp);

	//--------------------------------------------------------------------------
	// build ArgumentList op
	//--------------------------------------------------------------------------

	OpBase *argument_list = NewArgumentListOp(embedded_plan, arguments);
	// TODO: After refactoring the execution-plan freeing mechanism, bind the
	// ArgumentList op to the outer-scope plan (plan), and change the condition
	// for Unwind's 'free_rec' field to be whether the child plan is different
	// from the plan the Unwind is binded to.

	// add the op as a child of the unwind operation
	ExecutionPlan_AddOp(unwind, argument_list);

	// update the root of the (currently empty) embedded plan
	ExecutionPlan_UpdateRoot(embedded_plan, unwind);

	// build the execution-plan of the body of the clause
	AST *orig_ast = QueryCtx_GetAST();
	QueryCtx_SetAST(ast);
	ExecutionPlan_PopulateExecutionPlan(embedded_plan);
	QueryCtx_SetAST(orig_ast);

	// free the artificial body array (not its components)
	array_free(clauses);
	array_free(arguments);

	// connect the embedded plan to the Foreach op
	ExecutionPlan_AddOp(foreach, embedded_plan->root);
}

OpBase *ExecutionPlan_BuildOpsFromPath
(
	ExecutionPlan *plan,
	const char **bound_vars,
	const cypher_astnode_t *node
) {
	// initialize an ExecutionPlan that shares this plan's Record mapping
	ExecutionPlan *match_stream_plan = ExecutionPlan_NewEmptyExecutionPlan();
	match_stream_plan->record_map = plan->record_map;

	// if we have bound variables, build an Argument op that represents them
	if(bound_vars) match_stream_plan->root = NewArgumentOp(match_stream_plan,
															   bound_vars);

	AST *ast = QueryCtx_GetAST();
	// build a temporary AST holding a MATCH clause
	cypher_astnode_type_t type = cypher_astnode_type(node);

	// the AST node we're building a mock MATCH clause for will be a path
	// if we're converting a MERGE clause or WHERE filter, and we must build
	// and later free a CYPHER_AST_PATTERN node to contain it
	// if instead we're converting an OPTIONAL MATCH, the node is itself a MATCH clause
	// and we will reuse its CYPHER_AST_PATTERN node rather than building a new one
	bool node_is_path = (type == CYPHER_AST_PATTERN_PATH || type == CYPHER_AST_NAMED_PATH);
	AST *match_stream_ast = AST_MockMatchClause(ast, (cypher_astnode_t *)node, node_is_path);

	//--------------------------------------------------------------------------
	// build plan's query graph
	//--------------------------------------------------------------------------

	// extract pattern from holistic query graph
	const cypher_astnode_t **match_clauses = AST_GetClauses(match_stream_ast, CYPHER_AST_MATCH);
	ASSERT(array_len(match_clauses) == 1);

	const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match_clauses[0]);
	array_free(match_clauses);
	QueryGraph *sub_qg = QueryGraph_ExtractPatterns(plan->query_graph, &pattern, 1);
	match_stream_plan->query_graph = sub_qg;

	ExecutionPlan_PopulateExecutionPlan(match_stream_plan);

	AST_MockFree(match_stream_ast, node_is_path);
	QueryCtx_SetAST(ast); // reset the AST

	// associate all new ops with the correct ExecutionPlan and QueryGraph
	OpBase *match_stream_root = match_stream_plan->root;
	ExecutionPlan_BindOpsToPlan(plan, match_stream_root);

	// NULL-set map shared between the match_stream_plan and the overall plan
	match_stream_plan->record_map = NULL;

	// free the temporary plan
	ExecutionPlan_Free(match_stream_plan);

	return match_stream_root;
}

void ExecutionPlanSegment_ConvertClause
(
	GraphContext *gc,
	AST *ast,
	ExecutionPlan *plan,
	const cypher_astnode_t *clause
) {
	cypher_astnode_type_t t = cypher_astnode_type(clause);
	// because 't' is set using the offsetof() call
	// it cannot be used in switch statements
	if(t == CYPHER_AST_MATCH) {
		buildMatchOpTree(plan, ast, clause);
	} else if(t == CYPHER_AST_CALL) {
		buildCallOp(ast, plan, clause);
	} else if(t == CYPHER_AST_CREATE) {
		_buildCreateOp(gc, ast, plan, clause);
	} else if(t == CYPHER_AST_UNWIND) {
		_buildUnwindOp(plan, clause);
	} else if(t == CYPHER_AST_MERGE) {
		buildMergeOp(plan, ast, clause, gc);
	} else if(t == CYPHER_AST_SET || t == CYPHER_AST_REMOVE) {
		_buildUpdateOp(gc, plan, clause);
	} else if(t == CYPHER_AST_DELETE) {
		_buildDeleteOp(plan, clause);
	} else if(t == CYPHER_AST_RETURN) {
		// converting a RETURN clause can create multiple operations.
		buildReturnOps(plan, clause);
	} else if(t == CYPHER_AST_WITH) {
		// converting a WITH clause can create multiple operations.
		buildWithOps(plan, clause);
	} else if(t == CYPHER_AST_FOREACH) {
		_buildForeachOp(plan, clause, gc);
	} else if(t == CYPHER_AST_CALL_SUBQUERY) {
		buildCallSubqueryPlan(plan, clause);
	} else if(t == CYPHER_AST_LOAD_CSV) {
		_buildLoadCSVOp(plan, clause);
	} else {
		assert(false && "unhandeled clause");
	}
}

