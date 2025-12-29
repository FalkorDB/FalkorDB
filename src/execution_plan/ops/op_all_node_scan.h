/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"
#include "../../util/datablock/datablock_iterator.h"

// AllNodesScan
// scans through all nodes in the graph
typedef struct {
	OpBase op;                // op base
	const char *alias;        // alias of the node being scanned
	uint64_t progress;        // number of nodes processed thus far
	uint64_t node_count;      // number of nodes in graph
	uint32_t nodeRecIdx;      // position of node in record
	DataBlockIterator *iter;  // node iterator
	Record child_record;      // the record this op acts on if it is not a tap
} AllNodeScan;

// create a new all node scan operation
OpBase *NewAllNodeScanOp
(
	const ExecutionPlan *plan,
	const char *alias
);

