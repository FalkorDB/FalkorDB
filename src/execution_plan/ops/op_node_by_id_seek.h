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
#include "../execution_plan.h"
#include "../../graph/graph.h"
#include "../../util/range/unsigned_range.h"

#define ID_RANGE_UNBOUND -1

/* Node by ID seek locates an entity by its ID */
typedef struct {
	OpBase op;
	Graph *g;               // Graph object.
	Record child_record;    // The Record this op acts on if it is not a tap.
	const char *alias;      // Alias of the node being scanned by this op.
	NodeID currentId;       // Current ID fetched.
	NodeID minId;           // Min ID to fetch.
	NodeID maxId;           // Max ID to fetch.
	int nodeRecIdx;         // Position of entity within record.
} NodeByIdSeek;

OpBase *NewNodeByIdSeekOp(const ExecutionPlan *plan, const char *alias, UnsignedRange *id_range);

