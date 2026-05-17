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

// builds a linear chain of scan and traversal operations for a single
// pattern (sub-graph) extracted from the query graph
//
// starting from a root scan operation (label scan or all-node scan), one
// traversal op is prepended per algebraic expression derived from the pattern,
// forming a producer chain: scan -> traverse -> traverse -> ...
//
// return the topmost operation of the constructed chain, or NULL when the
// pattern needs no traversal (i.e. it is a single, already-bound
// label-free node)
OpBase *ExecutionPlan_ProcessPattern
(
	GraphContext *gc,     // graph context
	ExecutionPlan *plan,  // execution plan that will own the created operations
	QueryGraph *qg,       // full query graph for the entire query
	FT_FilterNode *ft,    // filter tree for the query, forwarded to expression
						  // ordering so selective filters are applied early
	bool optional,        // pattern is part of an OPTIONAL MATCH
	QueryGraph *pattern   // pattern Sub-graph describing the specific pattern
						  // to build ops for
) {
	Graph *g = GraphContext_GetGraph (gc) ;
	rax *bound_vars = plan->record_map ;

	//--------------------------------------------------------------------------
	// Early-exit: a single, optional node that is already bound needs
	// no scan or traversal
	//--------------------------------------------------------------------------

	if (optional && QueryGraph_EdgeCount (pattern) == 0) {
		QGNode *n = pattern->nodes [0] ;
		bool is_bound = raxFind (bound_vars, (unsigned char *)n->alias,
				strlen(n->alias)) != raxNotFound ;
		if (is_bound) {
			return NULL ;
		}
	}

	//--------------------------------------------------------------------------
	// build and order algebraic expressions for the pattern
	//--------------------------------------------------------------------------

	AlgebraicExpression **exps = AlgebraicExpression_FromQueryGraph (pattern) ;
	uint expCount = arr_len (exps) ;

	// reorder exps, to the most performant arrangement of evaluation
	orderExpressions (qg, exps, &expCount, ft, bound_vars);

	//--------------------------------------------------------------------------
	// create the leaf scan operation for the source node
	//--------------------------------------------------------------------------

	OpBase *root = NULL ;
	OpBase *tail = NULL ;

	// create a SCAN operation that will be the tail of the traversal chain
	QGNode *src = QueryGraph_GetNodeByAlias (qg,
			AlgebraicExpression_Src (exps[0])) ;

	uint label_count = QGNode_LabelCount (src) ;

	if (label_count > 0) {
		// the source node carries a label — peel it off the first expression
        // and turn it into a label-scan op
		AlgebraicExpression *ae_src =
			AlgebraicExpression_RemoveSource (&exps[0]) ;

		ASSERT (AlgebraicExpression_DiagonalOperand (ae_src, 0)) ;

		const char *label = AlgebraicExpression_Label (ae_src) ;
		const char *alias = AlgebraicExpression_Src (ae_src) ;

		ASSERT (label != NULL) ;
		ASSERT (alias != NULL) ;

		// resolve the internal label ID; remain GRAPH_UNKNOWN_LABEL if the
		// label does not exist in the schema (query will yield no results)
		int label_id = GRAPH_UNKNOWN_LABEL ;
		Schema *s = GraphContext_GetSchema (gc, label, SCHEMA_NODE) ;
		if (s != NULL) {
			label_id = Schema_GetID (s) ;
		}

		NodeScanCtx *ctx = NodeScanCtx_New (alias, label, label_id, src);
		root = tail = NewNodeByLabelScanOp (plan, ctx) ;

		AlgebraicExpression_Free (ae_src) ;
	} else {
		// no label — scan all nodes and filter downstream
		root = tail = NewAllNodeScanOp (plan, src->alias) ;

		// for edge-free patterns the first expression source has already been
		// consumed by the scan op; discard it to avoid a double-free
		if (arr_len (pattern->edges) == 0) {
			AlgebraicExpression_Free (
					AlgebraicExpression_RemoveSource (&exps [0])) ;
		}
	}

	//--------------------------------------------------------------------------
	// prepend one traversal operation per algebraic expression
	//--------------------------------------------------------------------------

	for (int j = 0 ; j < expCount ; j++) {
		AlgebraicExpression *exp = exps [j] ;

		// skip expressions that were fully consumed during scan construction
		if (AlgebraicExpression_OperandCount (exp) == 0) {
			continue ;
		}

		// determine whether this expression represents a variable-length edge
		QGEdge *edge = NULL;
		const char *edge_alias = AlgebraicExpression_Edge (exp) ;
		if (edge_alias) {
			edge = QueryGraph_GetEdgeByAlias(qg, edge_alias) ;
		}

		bool is_var_len = edge &&
			(QGEdge_VariableLength (edge) || !QGEdge_SingleHop (edge)) ;

		if (is_var_len) {
			if (QGEdge_IsShortestPath (edge)) {
				// allShortestPaths requires both endpoints to be already bound
				const char *src_alias  = QGNode_Alias (QGEdge_Src (edge)) ;
				const char *dest_alias = QGNode_Alias (QGEdge_Dest (edge)) ;

				bool src_bounded =
					raxFind (bound_vars, (unsigned char *)src_alias,
							strlen (src_alias)) != raxNotFound ;

				bool dest_bounded =
					raxFind (bound_vars, (unsigned char *)dest_alias,
							strlen (dest_alias)) != raxNotFound ;

				if (!src_bounded || !dest_bounded) {
					ErrorCtx_SetError (EMSG_ALLSHORTESTPATH_SRC_DST_RESLOVED) ;
				}
			}
			root = NewCondVarLenTraverseOp (plan, g, exp) ;
		} else {
			root = NewCondTraverseOp (plan, g, exp) ;
		}

		// link the new traversal op above the current top of the chain
		ExecutionPlan_AddOp (root, tail) ;
		tail = root;
	}

	// expression pointers have been transferred to their respective ops;
	// only the array wrapper itself needs to be freed
	arr_free (exps) ;

	return root ;
}

static OpBase *_ExecutionPlan_ProcessQueryGraph
(
	ExecutionPlan *plan,
	bool optional,
	QueryGraph *qg,
	AST *ast
) {
	OpBase *root = NULL ;
	GraphContext *gc = QueryCtx_GetGraphCtx () ;

	// build the full FilterTree for this AST
	// so that we can order traversals properly
	FT_FilterNode *ft = AST_BuildFilterTree (ast) ;
	QueryGraph **cc = QueryGraph_ConnectedComponents (qg) ;

	// if we have already constructed any ops
	// the plan's record map contains all variables bound at this time
	uint connectedComponentsCount = arr_len (cc) ;

	// keep track after all traversal operations along a pattern
	OpBase **streams = arr_new (OpBase *, 1) ;
	for (uint i = 0 ; i < connectedComponentsCount ; i++) {
		QueryGraph *component = cc [i] ;

		OpBase *stream =
			ExecutionPlan_ProcessPattern (gc, plan, qg, ft, optional, component) ;

		if (stream != NULL) {
			arr_append (streams, stream) ;
		}
	}

	uint n_streams = arr_len (streams) ;
	if (n_streams == 0) {
		goto cleanup ;
	}

	// if we have multiple graph components
	// the root operation is a cartesian product
	// each chain of traversals will be a child of this op

	root = streams [0] ;
	if (n_streams > 1) {
		root = NewCartesianProductOp (plan) ;
		for (uint i = 0 ; i < n_streams ; i++) {
			ExecutionPlan_AddOp (root, streams [i]) ;
		}
	}

cleanup:
	arr_free (streams) ;

	for (uint i = 0; i < connectedComponentsCount; i++) {
		QueryGraph_Free (cc [i]) ;
	}

	FilterTree_Free (ft) ;
	arr_free (cc) ;

	return root ;
}

void buildMatchOpTree
(
	ExecutionPlan *plan,
	AST *ast,
	const cypher_astnode_t *clause
) {
	rax *bound_vars = raxNew () ;

	// rather than cloning the record map, collect the bound variables
	// along with their parser-generated constant strings
	ExecutionPlan_BoundVariables (plan->root, bound_vars, plan) ;

	// collect the variable names from bound_vars to populate the
	// Argument op we will build
	const char **arguments = (const char **) raxValues (bound_vars) ;
	raxFree (bound_vars) ;

	bool optional = cypher_ast_match_is_optional (clause) ;

	const cypher_astnode_t *pattern = cypher_ast_match_get_pattern (clause) ;

	// collect the QueryGraph entities referenced in the clauses being converted
	QueryGraph *sub_qg =
		QueryGraph_ExtractPatterns (plan->query_graph, &pattern, 1) ;

	OpBase *stream =
		_ExecutionPlan_ProcessQueryGraph (plan, optional, sub_qg, ast) ;

	if (stream == NULL || ErrorCtx_EncounteredError ()) {
		goto cleanup ;
	}
	ASSERT (stream->parent == NULL) ;

	OpBase *op = stream ;

	if (optional) {
		OpBase *optional_op = NewOptionalOp (plan) ;
		ExecutionPlan_AddOp (optional_op, stream) ;
		op = optional_op ;
	}

	bool apply = plan->root != NULL &&
				 (optional || OpBase_Type (op) == OPType_CARTESIAN_PRODUCT) ;

	// connect a multi match streams
	if (apply) {
		OpBase *apply_op = NewApplyOp (plan) ;
		ExecutionPlan_UpdateRoot (plan, apply_op) ;
		ExecutionPlan_AddOp (apply_op, op) ;

		//----------------------------------------------------------------------
		// plant arguments
		//----------------------------------------------------------------------

		OpBase **taps = ExecutionPlan_CollectTaps (op) ;
		if (taps != NULL) {
			uint n = arr_len (taps) ;

			for (uint i = 0 ; i < n ; i++) {
				OpBase *tap = taps [i] ;
				OpBase *arg = NewArgumentOp (plan,  arguments) ;
				ExecutionPlan_AddOp (tap, arg) ;
			}

			arr_free (taps) ;
		}
	} else {
		ExecutionPlan_UpdateRoot (plan, op) ;
	}

	// build the FilterTree to model any WHERE predicates on these clauses
	// and place ops appropriately
	FT_FilterNode *sub_ft = AST_BuildFilterTreeFromClauses (ast, &clause, 1) ;
	if (sub_ft != NULL) {
		ExecutionPlan_PlaceFilterOps (plan, stream, sub_ft) ;
	}

	// clean up
cleanup:
	arr_free (arguments) ;
	QueryGraph_Free (sub_qg) ;
}

