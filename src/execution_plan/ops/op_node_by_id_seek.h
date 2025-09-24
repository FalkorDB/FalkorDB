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
#include "../../util/range/range.h"

// Node by ID seek locates an entity by its ID
typedef struct {
	OpBase op;
	Graph *g;                   // graph object
	Record child_record;        // the record this op acts on if it is not a tap
	const char *alias;          // alias of the node being scanned by this op
	RangeExpression *ranges;    // array of ID range expressions
	roaring64_bitmap_t *ids;    // IDs to scan
	roaring64_iterator_t *it;   // IDs iterator
	int nodeRecIdx;             // position of entity within record
} NodeByIdSeek;

// create a new NodeByIdSeek operation
OpBase *NewNodeByIdSeekOp
(
	const ExecutionPlan *plan,  // execution plan
	const char *alias,          // node alias
	RangeExpression *ranges     // ID range expressions
);

