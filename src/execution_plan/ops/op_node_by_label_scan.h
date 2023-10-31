/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "shared/scan_functions.h"
#include "../execution_plan.h"
#include "../../graph/graph.h"
#include "../../graph/entities/node.h"
#include "../../../deps/GraphBLAS/Include/GraphBLAS.h"
#include "../../util/range/unsigned_range.h"
#include "../../graph/rg_matrix/rg_matrix_iter.h"

// NodeByLabelScan, scans entire label

typedef struct {
	OpBase op;
	Graph *g;
	NodeScanCtx *n;             // label data of node being scanned
	unsigned int nodeRecIdx;    // node position within record
	UnsignedRange *id_range;    // ID range to iterate over
	RG_MatrixTupleIter iter;    // iterator over label matrix
	Record child_record;        // the Record this op acts on if it is not a tap
} NodeByLabelScan;

// creates a new NodeByLabelScan operation
OpBase *NewNodeByLabelScanOp
(
	ExecutionPlan *plan,
	NodeScanCtx *n
);

// transform a simple label scan to perform additional range query over the label  matrix
void NodeByLabelScanOp_SetIDRange
(
	NodeByLabelScan *op,
	UnsignedRange *id_range
);

