/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "../ops/ops.h"
#include "../../query_ctx.h"
#include "../execution_plan.h"
#include "../../errors/errors.h"
#include "execution_plan_util.h"
#include "execution_plan_modify.h"
#include "execution_plan_construct.h"
#include "../../util/rax_extensions.h"
#include "../optimizations/optimizations.h"
#include "../../ast/ast_build_filter_tree.h"

// build execution-plan operations which resolves traversal pattern
static OpBase* _processPattern
(
	GraphContext *gc,       // graph context
	ExecutionPlan *plan,    // execution plan
	QueryGraph *qg,         // pattern
	rax *bound_vars,        // bound variables
	FT_FilterNode *filters  // filters
) {
	ASSERT(gc         != NULL);
	ASSERT(qg         != NULL);
	ASSERT(plan       != NULL);
	ASSERT(bound_vars != NULL);

	OpBase *child  = NULL;
	OpBase *parent = NULL;
	uint edge_count = array_len(qg->edges);

	if(edge_count == 0) {
		// if there are no edges in the pattern, we only need a node scan
		// if no labels are introduced, and the var is bound, don't build
		// a traversal
		QGNode *n = qg->nodes[0];
		if(raxFind(bound_vars, (unsigned char *)n->alias, strlen(n->alias))
				!= raxNotFound && QGNode_LabelCount(n) == 0) {
			return NULL;
		}
	}

	// build an algebraic expression(s) for the current pattern
	AlgebraicExpression **exps = AlgebraicExpression_FromQueryGraph(qg);
	uint expCount = array_len(exps);

	// reorder exps, to the most performant arrangement of evaluation
	orderExpressions(qg, exps, &expCount, filters, bound_vars);

	// create SCAN operation that will be the tail of the traversal chain
	QGNode *src = QueryGraph_GetNodeByAlias(qg,
			AlgebraicExpression_Src(exps[0]));

	uint label_count = QGNode_LabelCount(src);
	if(label_count > 0) {
		AlgebraicExpression *ae_src =
			AlgebraicExpression_RemoveSource(&exps[0]);
		ASSERT(AlgebraicExpression_DiagonalOperand(ae_src, 0));

		const char *label = AlgebraicExpression_Label(ae_src);
		const char *alias = AlgebraicExpression_Src(ae_src);
		ASSERT(label != NULL);
		ASSERT(alias != NULL);

		int label_id = GRAPH_UNKNOWN_LABEL;
		Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
		if(s != NULL) label_id = Schema_GetID(s);

		// resolve source node by performing label scan
		NodeScanCtx *ctx = NodeScanCtx_New((char *)alias, (char *)label,
				label_id, src);
		parent = child = NewNodeByLabelScanOp(plan, ctx);

		// first operand has been converted into a label scan op
		AlgebraicExpression_Free(ae_src);
	} else {
		parent = child = NewAllNodeScanOp(plan, src->alias);
		// free expression source
		// in-case there are additional patterns to traverse
		if(array_len(qg->edges) == 0) {
			AlgebraicExpression_Free(
					AlgebraicExpression_RemoveSource(&exps[0]));
		}
	}

	// for each expression, build the appropriate traversal operation
	for(int j = 0; j < expCount; j++) {
		AlgebraicExpression *exp = exps[j];

		// empty expression, already freed
		if(AlgebraicExpression_OperandCount(exp) == 0) continue;

		QGEdge *edge = NULL;
		if(AlgebraicExpression_Edge(exp)) {
			edge = QueryGraph_GetEdgeByAlias(qg, AlgebraicExpression_Edge(exp));
		}

		if(edge && (QGEdge_VariableLength(edge) || !QGEdge_SingleHop(edge))) {
			if(QGEdge_IsShortestPath(edge)) {
				// edge is part of a shortest-path
				// MATCH allShortestPaths((a)-[*..]->(b))
				// validate both edge ends are bounded
				const char *src_alias  = QGNode_Alias(QGEdge_Src(edge));
				const char *dest_alias = QGNode_Alias(QGEdge_Dest(edge));
				bool src_bounded =
					raxFind(bound_vars, (unsigned char *)src_alias,
							strlen(src_alias)) != raxNotFound;
				bool dest_bounded =
					raxFind(bound_vars, (unsigned char *)dest_alias,
							strlen(dest_alias)) != raxNotFound;

				// TODO: would be great if we can perform this validation
				// at AST validation time
				if(!src_bounded || !dest_bounded) {
					ErrorCtx_SetError(EMSG_ALLSHORTESTPATH_SRC_DST_RESLOVED);
				}
			}
			parent = NewCondVarLenTraverseOp(plan, gc->g, exp);
		} else {
			parent = NewCondTraverseOp(plan, gc->g, exp);
		}

		// insert the new traversal op at the root of the chain
		ExecutionPlan_AddOp(parent, child);
		child = parent;
	}

	// free the expressions array
	// as its parts have been converted into operations
	array_free(exps);

	return parent;
}

// build execution plan where multiple branches are joined together
// by a cartesian product operation
// returns a cartesian product operation
static OpBase *_BuildCartesianProduct
(
	GraphContext *gc,         // graph context
	ExecutionPlan *plan,      // execution plan to build
	QueryGraph **components,  // traversal patterns
	uint n,                   // number of patterns
	rax *bound_vars,          // bound variables
	FT_FilterNode *ft         // filters
) {
	ASSERT(n          >  1);
	ASSERT(gc         != NULL);
	ASSERT(plan       != NULL);
	ASSERT(components != NULL);
	ASSERT(bound_vars != NULL);

	// creat the cartesian product operation
	OpBase *cp = NewCartesianProductOp(plan);

	for(uint i = 0; i < n; i++) {
		QueryGraph *p  = components[i];
		OpBase *branch = _processPattern(gc, plan, p, bound_vars, ft);

		// it is possible to get a NULL branch
		// MATCH (a) WITH a MATCH (a) RETURN a
		if(branch != NULL) ExecutionPlan_AddOp(cp, branch);
	}

	//--------------------------------------------------------------------------
	// cartesian product without any branches
	//--------------------------------------------------------------------------
	if(unlikely(OpBase_ChildCount(cp) == 0)) {
		// MATCH (a), (b) WITH a, b MATCH (a), (b) RETURN *
		OpBase_Free(cp);
		return NULL;
	}

	OpBase *root = cp;  // return value

	//--------------------------------------------------------------------------
	// cartesian product with a single branch is redundant
	//--------------------------------------------------------------------------
	if(unlikely(OpBase_ChildCount(cp) == 1)) {
		// remove the cartesian product operation
		root = OpBase_GetChild(cp, 0);
		ExecutionPlan_RemoveOp(plan, cp);
		OpBase_Free(cp);
	}

	return root;
}

// returns root to sub execution plan traversing MATCH pattern
static OpBase *_ExecutionPlan_ProcessQueryGraph
(
	ExecutionPlan *plan,  // plan to associate operations with
	QueryGraph *qg,       // traversal patterns
	AST *ast              // AST
) {
	GraphContext *gc = QueryCtx_GetGraphCtx();

	// build the full FilterTree for this AST
	// so that we can order traversals properly
	FT_FilterNode *ft = AST_BuildFilterTree(ast);

	// compute pattern connected components
	QueryGraph **cc = QueryGraph_ConnectedComponents(qg);
	uint n = array_len(cc);

	// if we have already constructed any ops
	// the plan's record map contains all variables bound at this time
	rax *bound_vars = plan->record_map;

	// if we have multiple graph components
	// the root operation is a cartesian product
	// each chain of traversals will be a child of this op
	OpBase *branch = NULL;
	if(n > 1) {
		branch = _BuildCartesianProduct(gc, plan, cc, n, bound_vars, ft);
	} else {
		branch = _processPattern(gc, plan, cc[0], bound_vars, ft);
	}

	array_free(cc);
	FilterTree_Free(ft);

	return branch;
}

// build traverse operation to resolve OPTIONAL pattern
static void _buildOptionalMatchOps
(
	ExecutionPlan *plan,            // plan to expand
	AST *ast,                       // AST
	const cypher_astnode_t *clause  // OPTIONAL MATCH clause
) {
	const char **arguments = NULL;
	OpBase *optional = NewOptionalOp(plan);
	rax *bound_vars = NULL;

	// the root will be non-null unless the first clause is an OPTIONAL MATCH
	if(plan->root) {
		// collect the variables that are bound at this point
		bound_vars = raxNew();
		// rather than cloning the record map
		// collect the bound variables along with their
		// parser-generated constant strings
		ExecutionPlan_BoundVariables(plan->root, bound_vars, plan);
		// collect the variable names from bound_vars to populate
		// the Argument op we will build
		arguments = (const char **)raxValues(bound_vars);
		raxFree(bound_vars);
	}

	// build the new Match stream and add it to the Optional stream
	OpBase *match_stream = ExecutionPlan_BuildOpsFromPath(plan, arguments,
			clause);
	array_free(arguments);
	ExecutionPlan_AddOp(optional, match_stream);

	// the root will be non-null unless the first clause is an OPTIONAL MATCH
	if(plan->root) {
		// create an Apply operator and make it the new root
		OpBase *apply_op = NewApplyOp(plan);
		ExecutionPlan_UpdateRoot(plan, apply_op);

		// create an Optional op and add it as an Apply child
		// as a right-hand stream
		ExecutionPlan_AddOp(apply_op, optional);
	} else {
		// if no root has been set (OPTIONAL was the first clause)
		// set it to the Optional op
		ExecutionPlan_UpdateRoot(plan, optional);
	}
}

// connect branch to plan
static void _ExecutionPlan_ConnectBranch
(
	ExecutionPlan *plan,  // plan to connect branch to
	OpBase *branch        // branch to connect
) {
	ASSERT(plan   != NULL);
	ASSERT(branch != NULL);
	ASSERT(branch->plan == plan);

	// empty plan
	if(plan->root == NULL) {
		ExecutionPlan_UpdateRoot(plan, branch);
		return;
	}

	OPType t = OpBase_Type(branch);
	bool indirect = (t == OPType_OPTIONAL || t == OPType_CARTESIAN_PRODUCT);

	OpBase **taps = ExecutionPlan_CollectTaps(branch);
	uint n = array_len(taps);

	// connect via an APPLY operation
	if(indirect) {
		// collect bound variables
		rax *bound_vars = raxNew();
		ExecutionPlan_BoundVariables(plan->root, bound_vars, plan);
		const char **variables = (const char **)raxValues(bound_vars);
		raxFree(bound_vars);

		// introduce ARGUMENT operation to each tap
		for(uint i = 0; i < n; i++) {
			OpBase *tap = taps[i];
			OpBase *arg = NewArgumentOp(plan, variables);
			ExecutionPlan_AddOp(tap, arg);
		}

		// connect branch via an APPLY operation
		OpBase *apply = NewApplyOp(plan);
		ExecutionPlan_UpdateRoot(plan, apply);
		ExecutionPlan_AddOp(apply, branch);
	} else {
		ASSERT(n == 1);
		ExecutionPlan_UpdateRoot(plan, branch);
	}

	array_free(taps);
}

void buildMatchOpTree
(
	ExecutionPlan *plan,
	AST *ast,
	const cypher_astnode_t *clause
) {
	ASSERT(ast    != NULL);
	ASSERT(plan   != NULL);
	ASSERT(clause != NULL);

	// collect the QueryGraph entities referenced in the clause being converted
	const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(clause);
	QueryGraph *sub_qg =
		QueryGraph_ExtractPatterns(plan->query_graph, &pattern, 1);

	OpBase *branch = _ExecutionPlan_ProcessQueryGraph(plan, sub_qg, ast);
	if(unlikely(branch == NULL))    goto cleanup;
	if(ErrorCtx_EncounteredError()) goto cleanup;

	// add OPTIONAL operation if pattern is optional
	if(cypher_ast_match_is_optional(clause)) {
		OpBase *optional = NewOptionalOp(plan);
		ExecutionPlan_AddOp(optional, branch);
		branch = optional;
	}

	// connect branch to plan
	_ExecutionPlan_ConnectBranch(plan, branch);

	// build the FilterTree to model any WHERE predicates on the clause
	// and place filters appropriately
	FT_FilterNode *sub_ft = AST_BuildFilterTreeFromClauses(ast, &clause, 1);
	ExecutionPlan_PlaceFilterOps(plan, branch, NULL, sub_ft);

	// clean up
cleanup:
	QueryGraph_Free(sub_qg);
}

