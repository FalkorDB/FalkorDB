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
	OpBase **tail,          // [optional] tail of the traversal chain
	GraphContext *gc,       // graph context
	QueryGraph *qg,         // pattern
	ExecutionPlan *plan,    // execution plan
	rax *bound_vars,        // bound variables
	FT_FilterNode *filters  // filters
) {
	ASSERT(gc         != NULL);
	ASSERT(qg         != NULL);
	ASSERT(plan       != NULL);
	ASSERT(bound_vars != NULL);

	if(tail != NULL) *tail = NULL;

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

	if(tail != NULL) *tail = child;

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
static void _BuildCartesianProduct
(
	GraphContext *gc,         // graph context
	ExecutionPlan *plan,      // execution plan to build
	QueryGraph **components,  // traversal patterns
	uint n,                   // number of patterns
	AST *ast,                 // AST
	rax *bound_vars,          // bound variables
	FT_FilterNode *ft         // filters
) {
	ASSERT(n          >  1);
	ASSERT(gc         != NULL);
	ASSERT(ast        != NULL);
	ASSERT(plan       != NULL);
	ASSERT(components != NULL);
	ASSERT(bound_vars != NULL);

	bool bounded = false;
	const char **vars = NULL;

	// bound branch exists
	// MATCH (n) WITH n MATCH (a {v:n.v}), (b {x:n.x}) RETURN *
	if(plan->root != NULL) {
		vars    = (const char**)raxKeys(bound_vars);
		bounded = true;
	}

	// creat the cartesian product operation
	OpBase *cp = NewCartesianProductOp(plan);

	for(uint i = 0; i < n; i++) {
		OpBase *tail = NULL;
		QueryGraph *p = components[i];
		OpBase *branch = _processPattern(&tail, gc, p, plan, bound_vars, ft);

		// it is possible to get a NULL branch
		// MATCH (a) WITH a MATCH (a) RETURN a
		if(branch == NULL) continue;

		ExecutionPlan_AddOp(cp, branch);

		if(bounded) {
			OpBase *arg = NewArgumentOp(plan, vars);
			// connect branch as the right child of cartesian product
			// and connect the plan's root as the left child
			ASSERT(tail != NULL);
			ExecutionPlan_AddOp(tail, arg);
		}
	}

	OpBase *branch = cp;

	// cartesian product with a single branch is redundant
	if(unlikely(OpBase_ChildCount(cp) == 1)) {
		ASSERT(bounded == true);
		// remove the cartesian product and Argument operations
		branch = OpBase_GetChild(cp, 0);
		OpBase *arg = ExecutionPlan_LocateOp(branch, OPType_ARGUMENT);
		ASSERT(arg != NULL);

		ExecutionPlan_RemoveOp(plan, cp);
		ExecutionPlan_RemoveOp(plan, arg);

		OpBase_Free(cp);
		OpBase_Free(arg);

		// no need to bind, simply directly link branches
		bounded = false;
	}

	array_free(vars);

	// if there are already operations in the plan
	// then the plan's root becomes a "bound" branch
	// for the cartesian product
	if(bounded) {
		// connect root as the left child of Apply
		// and connect cartesian product as the right child
		OpBase *apply = NewApplyOp(plan);
		ExecutionPlan_UpdateRoot(plan, apply);
		ExecutionPlan_AddOp(apply, branch);
	} else {
		ExecutionPlan_UpdateRoot(plan, branch);
	}
}

static void _ExecutionPlan_ProcessQueryGraph
(
	ExecutionPlan *plan,
	QueryGraph *qg,
	AST *ast
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
	if(n  > 1) {
		_BuildCartesianProduct(gc, plan, cc, n, ast, bound_vars, ft);
	} else {
		OpBase *branch = _processPattern(NULL, gc, cc[0], plan, bound_vars, ft);
		if(branch) ExecutionPlan_UpdateRoot(plan, branch);
	}

	array_free(cc);
	FilterTree_Free(ft);
}

static void _buildOptionalMatchOps
(
	ExecutionPlan *plan,
	AST *ast,
	const cypher_astnode_t *clause
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
	OpBase *match_stream = ExecutionPlan_BuildOpsFromPath(plan, arguments, clause);
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

void buildMatchOpTree
(
	ExecutionPlan *plan,
	AST *ast,
	const cypher_astnode_t *clause
) {
	if(cypher_ast_match_is_optional(clause)) {
		_buildOptionalMatchOps(plan, ast, clause);
		return;
	}

	const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(clause);

	// collect the QueryGraph entities referenced in the clauses being converted
	QueryGraph *sub_qg =
		QueryGraph_ExtractPatterns(plan->query_graph, &pattern, 1);

	_ExecutionPlan_ProcessQueryGraph(plan, sub_qg, ast);
	if(ErrorCtx_EncounteredError()) goto cleanup;

	// build the FilterTree to model any WHERE predicates on these clauses
	// and place ops appropriately
	FT_FilterNode *sub_ft = AST_BuildFilterTreeFromClauses(ast, &clause, 1);
	ExecutionPlan_PlaceFilterOps(plan, plan->root, NULL, sub_ft);

	// clean up
cleanup:
	QueryGraph_Free(sub_qg);
}

