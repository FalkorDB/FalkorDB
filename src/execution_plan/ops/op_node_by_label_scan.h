/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"
#include "../../graph/graph.h"
#include "../../util/roaring.h"
#include "shared/scan_functions.h"
#include "../../util/range/range.h"
#include "../../graph/delta_matrix/delta_matrix_iter.h"

// NodeByLabelScan, scans entire label
typedef struct {
	OpBase op;
	Graph *g;                     // graph
	NodeScanCtx *n;               // label data of node being scanned
	unsigned int nodeRecIdx;      // node position within record
	RangeExpression *ranges;      // array of ID range expressions
	roaring64_bitmap_t *ids;      // resolved ids by filters
	roaring64_iterator_t *ID_it;  // ID iterator
	Delta_Matrix L;               // label matrix
	Delta_MatrixTupleIter iter;   // iterator over label matrix
	Record child_record;          // the record this op acts on if it is not a tap
} NodeByLabelScan;

// creates a new NodeByLabelScan operation
OpBase *NewNodeByLabelScanOp
(
	const ExecutionPlan *plan,
	NodeScanCtx *n
);

// transform label scan to perform additional range query over the label matrix
void NodeByLabelScanOp_SetIDRange
(
	NodeByLabelScan *op,
	RangeExpression *ranges  // ID range expressions
);

