/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
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

/* NodeByLabelScan, scans entire label. */

typedef struct {
	OpBase op;
	Graph *g;
	NodeScanCtx *n;             // Label data of node being scanned
	unsigned int nodeRecIdx;    // Node position within record
	UnsignedRange *id_range;    // ID range to iterate over
	RG_MatrixTupleIter iter;    // Iterator over label matrix
	Record child_record;        // The Record this op acts on if it is not a tap
} NodeByLabelScan;

/* Creates a new NodeByLabelScan operation */
OpBase *NewNodeByLabelScanOp
(
	const ExecutionPlan *plan,
	NodeScanCtx *n
);

/* Transform a simple label scan to perform additional range query over the label  matrix. */
void NodeByLabelScanOp_SetIDRange(NodeByLabelScan *op, UnsignedRange *id_range);

