/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

// Results populates the query's result-set
// the operation enforces the configured maximum result-set size
// if that limit been reached query execution terminate
// otherwise the current record is added to the result-set

#pragma once

#include "op.h"
#include "../execution_plan.h"
#include "../../resultset/resultset.h"

typedef struct {
	OpBase op;
	ResultSet *result_set;
	uint64_t result_set_size_limit;
} Results;

// creates a new Results operation
OpBase *NewResultsOp
(
	const ExecutionPlan *plan
);

