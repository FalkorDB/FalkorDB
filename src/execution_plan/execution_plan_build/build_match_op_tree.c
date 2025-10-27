/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
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

static OpBase *_ExecutionPlan_ProcessQueryGraph
(
	ExecutionPlan *plan,
	QueryGraph *qg,
	AST *ast
) {
	GraphContext *gc = QueryCtx_GetGraphCtx();

	// build the full FilterTree for this AST
	// so that we can order traversals properly
	FT_FilterNode *ft = AST_BuildFilterTree(ast);
	QueryGraph **connectedComponents = QueryGraph_ConnectedComponents(qg);

	// if we have already constructed any ops
	// the plan's record map contains all variables bound at this time
	uint connectedComponentsCount = array_len(connectedComponents);
	rax *bound_vars = plan->record_map;

	// if we have multiple graph components
	// the root operation is a cartesian product
	// each chain of traversals will be a child of this op
	OpBase *apply            = NULL;
	OpBase *cartesianProduct = NULL;
	if(connectedComponentsCount > 1) {
		cartesianProduct = NewCartesianProductOp(plan);
	}

	// keep track after all traversal operations along a pattern
	for(uint i = 0; i < connectedComponentsCount; i++) {
		QueryGraph *cc = connectedComponents[i];
		uint edge_count = array_len(cc->edges);
		OpBase *root = NULL; // the root of the traversal chain will be added to the ExecutionPlan
		OpBase *tail = NULL;

		if(edge_count == 0) {
			// if there are no edges in the component, we only need a node scan
			// if no labels are introduced, and the var is bound, don't build
			// a traversal
			QGNode *n = cc->nodes[0];
			if(raxFind(bound_vars, (unsigned char *)n->alias, strlen(n->alias))
					!= raxNotFound && QGNode_LabelCount(n) == 0) {
				continue;
			}
		}

		AlgebraicExpression **exps = AlgebraicExpression_FromQueryGraph(cc);
		uint expCount = array_len(exps);

		// reorder exps, to the most performant arrangement of evaluation
		orderExpressions(qg, exps, &expCount, ft, bound_vars);

		// create a SCAN operation that will be the tail of the traversal chain
		QGNode *src = QueryGraph_GetNodeByAlias (qg,
				AlgebraicExpression_Src (exps[0])) ;

		uint label_count = QGNode_LabelCount (src) ;

		if (label_count > 0) {
			AlgebraicExpression *ae_src =
				AlgebraicExpression_RemoveSource (&exps[0]) ;
			ASSERT(AlgebraicExpression_DiagonalOperand (ae_src, 0)) ;

			const char *label = AlgebraicExpression_Label (ae_src) ;
			const char *alias = AlgebraicExpression_Src (ae_src) ;
			ASSERT (label != NULL) ;
			ASSERT (alias != NULL) ;

			int label_id = GRAPH_UNKNOWN_LABEL ;
			Schema *s = GraphContext_GetSchema (gc, label, SCHEMA_NODE) ;
			if (s != NULL) {
				label_id = Schema_GetID(s);
			}

			// resolve source node by performing label scan
			NodeScanCtx *ctx = NodeScanCtx_New (alias, label, label_id, src);
			root = tail = NewNodeByLabelScanOp(plan, ctx);

			// first operand has been converted into a label scan op
			AlgebraicExpression_Free(ae_src);
		} else {
			root = tail = NewAllNodeScanOp(plan, src->alias);
			// free expression source
			// in-case there are additional patterns to traverse
			if(array_len(cc->edges) == 0) {
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
				edge =
					QueryGraph_GetEdgeByAlias(qg, AlgebraicExpression_Edge(exp));
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
				root = NewCondVarLenTraverseOp(plan, gc->g, exp);
			} else {
				root = NewCondTraverseOp(plan, gc->g, exp);
			}
			// insert the new traversal op at the root of the chain
			ExecutionPlan_AddOp(root, tail);
			tail = root;
		}

		// free the expressions array
		// as its parts have been converted into operations
		array_free(exps);

		if(cartesianProduct) {
			// we have multiple disjoint traversal chains
			// add each chain as a child under the Cartesian Product
			ExecutionPlan_AddOp(cartesianProduct, root);
		} else {
			// we've built the only necessary traversal chain
			// update the ExecutionPlan root
			ExecutionPlan_UpdateRoot(plan, root);
		}
	}

	if(cartesianProduct != NULL && OpBase_ChildCount(cartesianProduct) > 0) {
		if(plan->root != NULL) {
			// connect existing operations via an apply operation
			// these will become the apply's left handside
			// while the cartesian product will be its right handside
			apply = NewApplyOp(plan);
			ExecutionPlan_UpdateRoot(plan, apply);

			ExecutionPlan_AddOp(apply, cartesianProduct);
		} else {
			ExecutionPlan_UpdateRoot(plan, cartesianProduct);
		}

		if(apply != NULL) {
			rax *bound_args = raxNew();
			ExecutionPlan_BoundVariables(OpBase_GetChild(apply, 0), bound_args,
					plan);
			const char **arguments = (const char**)raxValues(bound_args);

			// add Argument op to each branch within the cartesian product
			for(int i = 0; i < OpBase_ChildCount(cartesianProduct); i++) {
				OpBase *child = OpBase_GetChild(cartesianProduct, i);

				// get to the tip of the current branch
				while(OpBase_ChildCount(child) > 0) {
					child = OpBase_GetChild(child, 0);
				}

				// add argument to the tip of the branch
				OpBase *arg = NewArgumentOp(plan, arguments);
				ExecutionPlan_AddOp(child, arg);
			}

			raxFree(bound_args);
			array_free(arguments);
		}
	} else if(cartesianProduct != NULL) {
		// cartesian product has no children
		// free it and return the plan root
		OpBase_Free(cartesianProduct);
		cartesianProduct = NULL;
	}

	for(uint i = 0; i < connectedComponentsCount; i++) {
		QueryGraph_Free(connectedComponents[i]);
	}

	FilterTree_Free(ft);
	array_free(connectedComponents);

	return (cartesianProduct != NULL) ? cartesianProduct : plan->root;
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

	// The root will be non-null unless the first clause is an OPTIONAL MATCH.
	if(plan->root) {
		// Collect the variables that are bound at this point.
		bound_vars = raxNew();
		// Rather than cloning the record map, collect the bound variables along with their
		// parser-generated constant strings.
		ExecutionPlan_BoundVariables(plan->root, bound_vars, plan);
		// Collect the variable names from bound_vars to populate the Argument op we will build.
		arguments = (const char **)raxValues(bound_vars);
		raxFree(bound_vars);
	}

	// Build the new Match stream and add it to the Optional stream.
	OpBase *match_stream = ExecutionPlan_BuildOpsFromPath(plan, arguments, clause);
	array_free(arguments);
	ExecutionPlan_AddOp(optional, match_stream);

	// The root will be non-null unless the first clause is an OPTIONAL MATCH.
	if(plan->root) {
		// Create an Apply operator and make it the new root.
		OpBase *apply_op = NewApplyOp(plan);
		ExecutionPlan_UpdateRoot(plan, apply_op);

		// Create an Optional op and add it as an Apply child as a right-hand stream.
		ExecutionPlan_AddOp(apply_op, optional);
	} else {
		// If no root has been set (OPTIONAL was the first clause), set it to the Optional op.
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

	OpBase *op =_ExecutionPlan_ProcessQueryGraph(plan, sub_qg, ast);
	if(ErrorCtx_EncounteredError()) goto cleanup;

	// build the FilterTree to model any WHERE predicates on these clauses
	// and place ops appropriately
	FT_FilterNode *sub_ft = AST_BuildFilterTreeFromClauses(ast, &clause, 1);
	if(sub_ft != NULL) {
		ExecutionPlan_PlaceFilterOps(plan, op, sub_ft);
	}

	// clean up
cleanup:
	QueryGraph_Free(sub_qg);
}

