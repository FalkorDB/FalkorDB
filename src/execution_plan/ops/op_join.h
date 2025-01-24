/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

// the OpJoin operation joins multiple sub queries
// records pass through as is
// all records must have the same structure
//
// MATCH (n)-[:R]->(m)
// RETURN n.v AS value
// SORT BY n.k
//
// UNION
//
// MATCH (n)-[:S]->(m)->[]->()
// RETURN m.v + n.v AS value

#pragma once

#include "op.h"
#include "../execution_plan.h"

typedef struct {
	OpBase op;
	OpBase *stream;  // current stream to pull from
	uint streamIdx;  // current stream index
} OpJoin;

OpBase *NewJoinOp
(
	const ExecutionPlan *plan
);

