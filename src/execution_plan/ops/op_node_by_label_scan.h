/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "GraphBLAS.h"
#include "../execution_plan.h"
#include "../../graph/graph.h"
#include "../../util/roaring.h"
#include "shared/scan_functions.h"
#include "shared/filter_functions.h"
#include "../../graph/entities/node.h"
#include "../../graph/rg_matrix/rg_matrix_iter.h"

/* NodeByLabelScan, scans entire label. */

typedef struct {
	OpBase op;
	Graph *g;
	NodeScanCtx *n;             // label data of node being scanned
	unsigned int nodeRecIdx;    // node position within record
	FilterExpression *filters;  // filters expressions applied to id e.g. ID(n) > 10
	roaring64_bitmap_t *ids;    // resolved ids by filters
	roaring64_iterator_t *it;   // id iterators
	RG_Matrix L;                // label matrix
	RG_MatrixTupleIter iter;    // iterator over label matrix
	Record child_record;        // the record this op acts on if it is not a tap
} NodeByLabelScan;

// creates a new NodeByLabelScan operation
OpBase *NewNodeByLabelScanOp
(
	const ExecutionPlan *plan,
	NodeScanCtx *n
);

void NodeByLabelScanOp_SetFilterID
(
	NodeByLabelScan *op,
	FilterExpression *filters
);
