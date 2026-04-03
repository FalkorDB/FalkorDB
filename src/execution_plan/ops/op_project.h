/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"
#include "../../arithmetic/arithmetic_expression.h"

typedef struct {
	OpBase op;
	Record r;                // input Record being read from (stored to free if we encounter an error).
	Record projection;       // record projected by this operation (stored to free if we encounter an error).
	AR_ExpNode **exps;       // projected expressions (including order exps).
	uint *record_offsets;    // record IDs corresponding to each projection (including order exps).
	bool singleResponse;     // when no child operations, return NULL after a first response.
	uint exp_count;          // number of projected expressions.
} OpProject;

// create a new projection operation
OpBase *NewProjectOp
(
	const ExecutionPlan *plan,  // op's plan
	AR_ExpNode **exps           // projection expression
);

// binds a Project op to an ExecutionPlan
void ProjectBindToPlan
(
	OpBase *opBase,            // op to bind
	const ExecutionPlan *plan  // plan to bind the op to
);

