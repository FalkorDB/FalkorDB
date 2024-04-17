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
#include "shared/filter_functions.h"

#define ID_RANGE_UNBOUND -1

/* Node by ID seek locates an entity by its ID */
typedef struct {
	OpBase op;
	Graph *g;                   // graph object
	Record child_record;        // the record this op acts on if it is not a tap
	const char *alias;          // alias of the node being scanned by this op
	FilterExpression *filters;  // filters expressions applied to id e.g. ID(n) > 10
	roaring64_bitmap_t *ids;    // resolved ids by filters
	roaring64_iterator_t *it;   // id iterators
	int nodeRecIdx;             // position of entity within record
} NodeByIdSeek;

OpBase *NewNodeByIdSeekOp
(
	const ExecutionPlan *plan,
	const char *alias,
	FilterExpression *filters
);

