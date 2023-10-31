/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"
#include "../../graph/graph.h"
#include "../../util/range/unsigned_range.h"

#define ID_RANGE_UNBOUND -1

// Node by ID seek locates an entity by its ID
typedef struct {
	OpBase op;
	Graph *g;             // graph object
	Record child_record;  // the Record this op acts on if it is not a tap
	const char *alias;    // alias of the node being scanned by this op
	NodeID currentId;     // current ID fetched
	NodeID minId;         // min ID to fetch
	NodeID maxId;         // max ID to fetch
	int nodeRecIdx;       // position of entity within record
} NodeByIdSeek;

OpBase *NewNodeByIdSeekOp
(
	ExecutionPlan *plan,
	const char *alias,
	UnsignedRange *id_range
);

