/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"
#include "../../graph/graph.h"
#include "../../graph/query_graph.h"
#include "../../graph/entities/node.h"
#include "../../util/datablock/datablock_iterator.h"

/* AllNodesScan
 * Scans entire graph */
typedef struct {
	OpBase op;
	const char *alias;          /* Alias of the node being scanned by this op. */
	uint nodeRecIdx;
	DataBlockIterator *iter;
	Record child_record;        /* The Record this op acts on if it is not a tap. */
} AllNodeScan;

OpBase *NewAllNodeScanOp(const ExecutionPlan *plan, const char *alias);

