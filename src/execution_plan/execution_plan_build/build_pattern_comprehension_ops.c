/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "../ops/ops.h"
#include "../../query_ctx.h"
#include "execution_plan_util.h"
#include "execution_plan_construct.h"
#include "../../util/rax_extensions.h"
#include "../../ast/ast_build_filter_tree.h"
#include "../execution_plan_build/execution_plan_modify.h"
#include "../../arithmetic/arithmetic_expression_construct.h"

// build pattern comprehension plan operations for example:
// RETURN [p = (n)-->() | p] AS ps
// Results
//     Project
//         Optional
//             Aggregate
//                 Conditional Traverse | (n)-[anon_0]->(anon_1)
//                     All Node Scan | (n)
//
// MATCH (n) RETURN [p = (n)-->() | p] AS ps
// Results
//     Project
//         Apply
//             All Node Scan | (n)
//             Optional
//                 Aggregate
//                     Conditional Traverse | (n)-[anon_1]->(anon_2)
//                         Argument

void buildPatternComprehensionOps
(
	ExecutionPlan *plan,
	OpBase *root,
	const cypher_astnode_t *ast
) {
	// validations
	ASSERT(plan != NULL);
	ASSERT(root != NULL);
	ASSERT(ast  != NULL);
	ASSERT(root->type == OPType_PROJECT || root->type == OPType_AGGREGATE);

	// search for pattern comprehension AST nodes
	// quickly return if none been found
	const cypher_astnode_t **pcs =
		AST_GetTypedNodes(ast, CYPHER_AST_PATTERN_COMPREHENSION);
	uint count = array_len(pcs);

	if(count == 0) {
		array_free(pcs);
		return;
	}

	// backup AST, restore at the end
	AST *prev_ast = QueryCtx_GetAST();
	QueryCtx_SetAST(plan->ast_segment);

	const char **arguments = NULL;
	if(root->childCount > 0) {
		// get the bound variable to use when building the traversal ops
		rax *bound_vars = raxNew();
		ExecutionPlan_BoundVariables(root->children[0], bound_vars,
			root->children[0]->plan);
		arguments = (const char **)raxValues(bound_vars);
		raxFree(bound_vars);
	}

	for (uint i = 0; i < count; i++) {
		OpBase *match_stream;
		const cypher_astnode_t *pc;
		const cypher_astnode_t *path;
		const cypher_astnode_t *eval_node;
		const cypher_astnode_t *predicate;

		pc = pcs[i];
		path = cypher_ast_pattern_comprehension_get_pattern(pc);

		// constuct sub execution-plan resolving path
		match_stream = ExecutionPlan_BuildOpsFromPath(plan, arguments, path);

		// construct evaluation expression
		// [(a)-[]->(z) | z.v] `z.v` is the evaluation expression
		eval_node = cypher_ast_pattern_comprehension_get_eval(pc);
		AR_ExpNode *eval_exp = AR_EXP_FromASTNode(eval_node);

		// collect evaluation results into an array using `collect`
		AR_ExpNode *collect_exp = AR_EXP_NewOpNode("collect", false, 1);
		collect_exp->op.children[0] = eval_exp;
		collect_exp->resolved_name = AST_ToString(pc);

		// add collect expression to an AGGREGATION Operation
		AR_ExpNode **exps = array_new(AR_ExpNode *, 1);
		array_append(exps, collect_exp);
		OpBase *aggregate = NewAggregateOp(plan, exps);
		ExecutionPlan_AddOp(aggregate, match_stream);

		// handle filters attached to pattern
		// [(a {v:1})-[]->(z) WHERE z.v = 2 | z.x]
		predicate = cypher_ast_pattern_comprehension_get_predicate(pc);
		if(predicate != NULL) {
			// build filter tree
			FT_FilterNode *filter_tree = NULL;
			AST_ConvertFilters(&filter_tree, predicate);

			if(!FilterTree_Valid(filter_tree)) {
				// Invalid filter tree structure, a compile-time error has been set.
				FilterTree_Free(filter_tree);
			} else {
				// Apply De Morgan's laws
				FilterTree_DeMorgan(&filter_tree);

				// place filters
				ExecutionPlan_PlaceFilterOps(plan, aggregate, NULL, filter_tree);
			}
		}

		// in case the execution-plan had child operations we need to combine
		// records coming out of our newly constucted aggregation with the rest
		// of the execution-plan, use APPLY to do so
		// otherwise (no children) RETURN [() | n.v] simply set `root`
		if(root->childCount > 0) {
			OpBase *apply_op = NewApplyOp(plan);
			ExecutionPlan_PushBelow(root->children[0], apply_op);
			ExecutionPlan_AddOp(apply_op, aggregate);
		} else {
			ExecutionPlan_AddOp(root, aggregate);
		}
	}

	// restore AST
	QueryCtx_SetAST(prev_ast);

	if(arguments != NULL) array_free(arguments);
	array_free(pcs);
}

// build pattern path plan operations for example:
// RETURN ()-->() AS ps
// Results
//     Project
//         Optional
//             Aggregate
//                 Conditional Traverse | (anon_0)-[anon_1]->(anon_2)
//                     All Node Scan | (anon_0)
//
// MATCH (n) RETURN (n)-->() AS ps
// Results
//     Project
//         Apply
//             All Node Scan | (n)
//             Optional
//                 Aggregate
//                     Conditional Traverse | (n)-[anon_1]->(anon_2)
//                         Argument

void buildPatternPathOps
(
	ExecutionPlan *plan,
	OpBase *root,
	const cypher_astnode_t *ast
) {
	// validations
	ASSERT(plan != NULL);
	ASSERT(root != NULL);
	ASSERT(ast  != NULL);
	ASSERT(root->type == OPType_PROJECT || root->type == OPType_AGGREGATE);

	// search for pattern comprehension AST nodes
	// quickly return if none been found
	const  cypher_astnode_t  **sps  =  AST_GetTypedNodes(ast,  CYPHER_AST_SHORTEST_PATH);
	const  cypher_astnode_t  **pps  =  AST_GetTypedNodes(ast,  CYPHER_AST_PATTERN_PATH);
	const  cypher_astnode_t  **pcs  =  AST_GetTypedNodes(ast,  CYPHER_AST_PATTERN_COMPREHENSION);

	uint  count      =  array_len(pps);
	uint  pcs_count  =  array_len(pcs);
	uint  sps_count  =  array_len(sps);

	// backup AST, restore at the end
	AST *prev_ast = QueryCtx_GetAST();
	QueryCtx_SetAST(plan->ast_segment);

	const char **arguments = NULL;
	if(root->childCount > 0) {
		// get the bound variable to use when building the traversal ops
		rax *bound_vars = raxNew();
		ExecutionPlan_BoundVariables(root->children[0], bound_vars,
			root->children[0]->plan);
		arguments = (const char **)raxValues(bound_vars);
		raxFree(bound_vars);
	}

	for (uint i = 0; i < count; i++) {
		OpBase *match_stream;
		const cypher_astnode_t *path = pps[i];
		
		// skip pattern paths declared within pattern comprehension
		// as these are already handeled by `buildPatternComprehensionOps`
		bool skip_path = false;
		for (uint j = 0; j < pcs_count; j++) {
			if(cypher_ast_pattern_comprehension_get_pattern(pcs[j]) == path) {
				skip_path = true;
				break;
			}
		}

		// if this pattern path is nested within a pattern comprehension
		// we already handled it `buildPatternComprehensionOps`
		if(skip_path) continue;

		// skip pattern paths declared within shortest path
		// as these were already handeled
		for (uint j = 0; j < sps_count; j++) {
			if(cypher_ast_shortest_path_get_path(sps[j]) == path) {
				skip_path = true;
				break;
			}
		}

		// if this pattern path is nexted within a shortest path
		// we already handeled it
		if(skip_path) continue;

		// construct sub execution-plan resolving path
		match_stream = ExecutionPlan_BuildOpsFromPath(plan, arguments, path);

		// count number of elements in path
		// construct a `topath` expression combining elements into a PATH object
		uint path_len = cypher_ast_pattern_path_nelements(path);
		AR_ExpNode *path_exp = AR_EXP_NewOpNode("topath", true, 1 + path_len);
		path_exp->op.children[0] =
			AR_EXP_NewConstOperandNode(SI_PtrVal((void *)path));
		for(uint j = 0; j < path_len; j ++) {
			path_exp->op.children[j + 1] =
				AR_EXP_FromASTNode(cypher_ast_pattern_path_get_element(path, j));
		}

		// we're require to return an ARRAY of paths, use `collect` to aggregate paths
		AR_ExpNode *collect_exp = AR_EXP_NewOpNode("collect", false, 1);
		collect_exp->op.children[0] = path_exp;
		collect_exp->resolved_name = AST_ToString(path);

		// constuct aggregation operation
		AR_ExpNode **exps = array_new(AR_ExpNode *, 1);
		array_append(exps, collect_exp);
		OpBase *aggregate = NewAggregateOp(plan, exps);
		ExecutionPlan_AddOp(aggregate, match_stream);

		// in case the execution-plan had child operations we need to combine
		// records coming out of our newly constucted aggregation with the rest
		// of the execution-plan, use APPLY to do so
		// otherwise (no children) RETURN () simply set `root`
		if(root->childCount > 0) {
			OpBase *apply_op = NewApplyOp(plan);
			ExecutionPlan_PushBelow(root->children[0], apply_op);
			ExecutionPlan_AddOp(apply_op, aggregate);
		} else {
			ExecutionPlan_AddOp(root, aggregate);
		}
	}

	// restore AST
	QueryCtx_SetAST(prev_ast);

	if(arguments != NULL) array_free(arguments);
	array_free(pps);
	array_free(pcs);
	array_free(sps);
}

