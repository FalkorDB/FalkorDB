/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../query_ctx.h"
#include "../ops/op_expand_into.h"
#include "../ops/op_node_by_label_scan.h"
#include "../ops/op_conditional_traverse.h"
#include "../execution_plan_build/execution_plan_util.h"
#include "../execution_plan_build/execution_plan_modify.h"
#include "../../arithmetic/algebraic_expression/utils.h"

// this optimization scans through each label-scan operation
// in case the node being scaned is associated with multiple labels
// e.g. MATCH (n:A:B) RETURN n
// we will prefer to scan through the label with the least amount of nodes
// for the above example if NNZ(A) < NNZ(B) we will want to iterate over A
//
// in-case this optimization changed the label scanned e.g. from A to B
// it will have to patch the following traversal removing B operand
// and adding A back
//
// consider MATCH (n:A:B)-[:R]->(m) RETURN m
// 
// Scan(A)
// Traverse B*R
//
// if we switch from Scan(A) to Scan(B)
// we will have to update the traversal pattern from B*R to A*R
//
// Scan(B)
// Traverse A*R

// expressions such as:
// MATCH (a:A)-[]->(b:B) RETURN a, b
//
// where both ends 'a' and 'b' are scored evenly by the compile time
// planner need to be tie breaked
// the opening expression should be the one with the least amount of entities
// associcated with it
//
// in case the planner determined that 'a' should be the starting point
// but nnz(A) > nnz(B) then we'll need to transpose the expression
// switching from:
//
// Conditional Traverse | (a)->(b:B)"
//     Node By Label Scan | (a:A)"
//
// Conditional Traverse | (b)<-(a:A)"
//     Node By Label Scan | (b:B)"
static void _transposeExpression
(
	NodeByLabelScan *scan
) {
	ASSERT(scan != NULL);

	OpBase *op = (OpBase*)scan;

	// only address plans where src isn't filtered
	// as we don't want to reposition the filter
	// it is likely that former logic determined that this is the right
	// node to begining traversal from
	OpBase *parent = op->parent;

	// expecting a traverse operation following the label scan
	// TODO: support variable length traversal
	if(OpBase_Type(parent) != OPType_CONDITIONAL_TRAVERSE) {
		return;
	}

	OpCondTraverse *traversal = (OpCondTraverse*)parent;

	// we shouldn't care about the following operation(s)
	// the scenario where a traversal is followed by a filter is only
	// possible when the filter is applicable to either:
	// both the src and the destination e.g. n.v = m.v
	// or both the destination and the edge are filtered e.g. m.v = e.v
	// a situation where only the destination node is filtered isn't possible
	// due to former ordering logic which should have choosen the destination
	// node as the 'opening" node for the traversal

	Graph               *g          = QueryCtx_GetGraph();
	const GraphContext  *gc         = QueryCtx_GetGraphCtx();
	const ExecutionPlan *plan       = op->plan;
	QueryGraph          *qg         = plan->query_graph;
	AlgebraicExpression *ae         = AlgebraicExpression_Clone(traversal->ae);
	NodeScanCtx         *scan_ctx   = scan->n;
	const char          *dest_alias = AlgebraicExpression_Dest(ae);

	// split the operands within the algebraic expression into two parts
	// 1. src labels - these are the leftmost operands
	// 2. dest labels - these are the rightmost operands
	// e.g.
	//
	// (:A:B:C)-[:R]->(:X:Y:Z)
	//
	// A * B * C * R * X * Y * Z
	// A, B & C are source labels
	// X, Y & Z are destination labels

	uint n;
	AlgebraicExpression **operands =
		AlgebraicExpression_CollectOperandsInOrder(ae, &n);

	uint src_ops_n  = 0;               // number of source labels
	uint dest_ops_n = 0;               // number of destination labels
	AlgebraicExpression *src_ops[n];   // source labels
	AlgebraicExpression *dest_ops[n];  // destination labels

	// populate source labels array
	// break upon first none diagonal operand
	// this denotes a relationship matrix
	for(int i = 0; i < n; i++) {
		AlgebraicExpression *operand = operands[i];
		if(AlgebraicExpression_Diagonal(operand) == true) {
			src_ops[src_ops_n++] = operand;
		} else {
			// none diagonal matrix, we're done
			break;
		}
	}

	// populate destination labels array
	// scan from end backwards, break upon first none diagonal operand
	// this denotes a relationship matrix
	for(int i = n-1; i >= 0; i--) {
		AlgebraicExpression *operand = operands[i];
		if(AlgebraicExpression_Diagonal(operand) == true) {
			dest_ops[dest_ops_n++] = operand;
		} else {
			// none diagonal matrix, we're done
			break;
		}
	}

	// free operands array
	free(operands);

	// return if destination isn't associcated with any labels
	if(dest_ops_n == 0) {
		return;
	}

	// determine nim number of entities for both source node and dest node
	// src_min  = MIN(NVALS(src_lbls))
	// dest_min = MIN(NVALS(dest_lbls))
	const char *min_lbl    = NULL;
	LabelID     min_lbl_id = GRAPH_NO_LABEL;

	// src_min is initialized with the label-scan operation's matrix nvlas
	uint64_t src_min  = Graph_LabeledNodeCount(g, scan_ctx->label_id);
	uint64_t dest_min = UINT64_MAX;

	//--------------------------------------------------------------------------
	// determine source label min entities
	//--------------------------------------------------------------------------

	for(uint i = 0; i < src_ops_n; i++) {
		const AlgebraicExpression *operand = src_ops[i];

		uint64_t entity_count = 0;
		const char *lbl = AlgebraicExpression_Label(operand);

		Schema *s = GraphContext_GetSchema(gc, lbl, SCHEMA_NODE);
		if(s != NULL) {
			LabelID lbl_id = Schema_GetID(s);
			entity_count = Graph_LabeledNodeCount(g, lbl_id);
		}

		// a new minimum found
		if(src_min > entity_count) {
			src_min = entity_count;
		}
	}

	//--------------------------------------------------------------------------
	// determine destination label with min entities
	//--------------------------------------------------------------------------

	for(uint i = 0; i < dest_ops_n; i++) {
		const AlgebraicExpression *operand = dest_ops[i];

		uint64_t entity_count = 0;
		LabelID lbl_id = GRAPH_UNKNOWN_LABEL;
		const char *lbl = AlgebraicExpression_Label(operand);

		Schema *s = GraphContext_GetSchema(gc, lbl, SCHEMA_NODE);
		if(s != NULL) {
			lbl_id = Schema_GetID(s);
			entity_count = Graph_LabeledNodeCount(g, lbl_id);
		}

		// a new minimum found
		if(dest_min > entity_count) {
			dest_min   = entity_count;
			min_lbl    = lbl;
			min_lbl_id = lbl_id;
		}
	}

	// check if we should replace the current label scan operation
	if(dest_min >= src_min) {
		// source will produce less entities, keep things as they are
		return;
	}

	// scanning destination entities will produce less entities
	// reverse traverse pattern
	// e.g.
	// (a)->(b)
	// will become
	// (b)<-(a)

	// add back source label matrix to traverse expression
	AlgebraicExpression *lhs =
		AlgebraicExpression_NewOperand(NULL, true, scan_ctx->alias,
				scan_ctx->alias, NULL, scan_ctx->label);

	// multiply to the left as we're adding back the old src
	ae = _AlgebraicExpression_MultiplyToTheLeft(lhs, ae);

	// remove migrated destination label matrix from traverse expression
	AlgebraicExpression *operand = NULL;
	bool found = AlgebraicExpression_LocateOperand(ae, &operand, NULL,
			dest_alias, dest_alias, NULL, min_lbl);
	ASSERT(found   == true);
	ASSERT(operand != NULL);

	AlgebraicExpression_RemoveOperand(&ae, operand);
	AlgebraicExpression_Free(operand);

	// transpose conditional traverse expression
	// as we're going in the reverse direction
	AlgebraicExpression_Transpose(&ae);

	// create a new LabelScan operation scanning the minimal destination label
	QGNode *dest_node = QueryGraph_GetNodeByAlias(qg, dest_alias);
	NodeScanCtx *ctx = NodeScanCtx_New(dest_alias, min_lbl, min_lbl_id,
			dest_node);

	// replace current label scan with new one
	OpBase *new_scan = NewNodeByLabelScanOp(plan, ctx);
	ExecutionPlan_ReplaceOp((ExecutionPlan*)plan, (OpBase*)scan,
			(OpBase*)new_scan);
	OpBase_Free((OpBase*)scan);

	// replace current traversal with new one
	OpBase *new_traversal = NewCondTraverseOp((ExecutionPlan*)plan, g, ae);
	ExecutionPlan_ReplaceOp((ExecutionPlan*)plan, (OpBase*)traversal,
			(OpBase*)new_traversal);
	OpBase_Free((OpBase*)traversal);

	return;
}

static void _costBaseLabelScan
(
	NodeByLabelScan *scan
) {
	ASSERT(scan != NULL);

	Graph       *g     = QueryCtx_GetGraph();
	OpBase      *op    = (OpBase*)scan;
	QueryGraph *qg     = op->plan->query_graph;
	NodeScanCtx *n_ctx = scan->n;

	// see if scanned node has multiple labels
	const char *node_alias = n_ctx->alias;
	QGNode *n = n_ctx->n;

	// return if node has only one label
	uint label_count = QGNode_LabelCount(n);
	ASSERT(label_count >= 1);
	if(label_count == 1) {
		return;
	}

	// node has multiple labels
	// find label with minimum entities
	int min_label_id = n_ctx->label_id;
	const char *min_label_str = n_ctx->label;
	uint64_t min_nnz =(uint64_t) Graph_LabeledNodeCount(g, n_ctx->label_id);

	for(uint i = 0; i < label_count; i++) {
		uint64_t nnz;
		int label_id = QGNode_GetLabelID(n, i);
		nnz = Graph_LabeledNodeCount(g, label_id);
		if(min_nnz > nnz) {
			// update minimum
			min_nnz       = nnz;
			min_label_id  = label_id;
			min_label_str = QGNode_GetLabel(n, i);
		}
	}

	// scanned label has the minimum number of entries
	// no switching required
	if(min_label_id == n_ctx->label_id) {
		return;
	}

	// patch following traversal, skip filters
	OpBase *parent = op->parent;
	while(OpBase_Type(parent) == OPType_FILTER) parent = parent->parent;
	OPType t = OpBase_Type(parent);
	ASSERT(t == OPType_CONDITIONAL_TRAVERSE || t == OPType_EXPAND_INTO);

	AlgebraicExpression *ae = NULL;
	if(t == OPType_CONDITIONAL_TRAVERSE) {
		// GRAPH.EXPLAIN g "match (n:B:A:C)-[]->() RETURN n"
		// 1) "Results"
		// 2) "    Project"
		// 3) "        Conditional Traverse | (n:B:C)->(@anon_0)"
		// 4) "            Node By Label Scan | (n:A)"
		OpCondTraverse *op_traverse = (OpCondTraverse*)parent;
		ae = op_traverse->ae;
	} else {
		// GRAPH.EXPLAIN g "MATCH (n:B:A:C) RETURN n"
		// 1) "Results"
		// 2) "    Project"
		// 3) "        Expand Into | (n:A:C)->(n:A:C)"
		// 4) "            Node By Label Scan | (n:B)"
		OpExpandInto *op_expand = (OpExpandInto*)parent;
		ae = op_expand->ae;
	}

	AlgebraicExpression *operand;
	const char *row_domain    = n_ctx->alias;
	const char *column_domain = n_ctx->alias;

	// locate the operand corresponding to the about to be replaced label
	// in the parent operation (conditional traverse)
	bool found = AlgebraicExpression_LocateOperand(ae, &operand, NULL,
			row_domain, column_domain, NULL, min_label_str);
	ASSERT(found == true);

	// create a replacement operand for the migrated label matrix
	AlgebraicExpression *replacement = AlgebraicExpression_NewOperand(NULL,
			true, AlgebraicExpression_Src(operand),
			AlgebraicExpression_Dest(operand), NULL, n_ctx->label);

	// swap current label with minimum label
	n_ctx->label    = min_label_str;
	n_ctx->label_id = min_label_id;

	_AlgebraicExpression_InplaceRepurpose(operand, replacement);
}

void costBaseLabelScan
(
	ExecutionPlan *plan
) {
	ASSERT(plan != NULL);

	// collect all label scan operations
	OPType t = OPType_NODE_BY_LABEL_SCAN;
	OpBase **label_scan_ops = ExecutionPlan_CollectOpsMatchingTypes(plan->root,
			&t ,1);

	// for each label scan operation try to optimize scanned label
	uint op_count = array_len(label_scan_ops);
	for(uint i = 0; i < op_count; i++) {
		NodeByLabelScan *label_scan = (NodeByLabelScan*)label_scan_ops[i];
		_costBaseLabelScan(label_scan);
		_transposeExpression(label_scan);
	}

	array_free(label_scan_ops);
}

