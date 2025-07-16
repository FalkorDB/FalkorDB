/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "../../query_ctx.h"
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
// it will have to patch the following conditional traversal removing B operand
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

static void _optimizeLabelScan(NodeByLabelScan *scan) {
	ASSERT(scan != NULL);	

	Graph       *g     =  QueryCtx_GetGraph();
	OpBase      *op    =  (OpBase*)scan;
	QueryGraph *qg     =  op->plan->query_graph;
	NodeScanCtx *n_ctx =  scan->n;

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
	ASSERT(OpBase_Type(parent) == OPType_CONDITIONAL_TRAVERSE);

	OpCondTraverse *op_traverse = (OpCondTraverse*)parent;
	AlgebraicExpression *ae = op_traverse->ae;
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

void optimizeLabelScan(ExecutionPlan *plan) {
	ASSERT(plan != NULL);

	// collect all label scan operations
	OPType t = OPType_NODE_BY_LABEL_SCAN;
	OpBase **label_scan_ops = ExecutionPlan_CollectOpsMatchingTypes(plan->root,
			&t ,1);

	// for each label scan operation try to optimize scanned label
	uint op_count = array_len(label_scan_ops);
	for(uint i = 0; i < op_count; i++) {
		NodeByLabelScan *label_scan = (NodeByLabelScan*)label_scan_ops[i];
		_optimizeLabelScan(label_scan);
	}
	array_free(label_scan_ops);
}
