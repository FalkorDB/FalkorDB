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
#include "op_argument.h"
#include "../execution_plan.h"

/* The Apply op has a bound left-hand branch
 * and a right-hand branch that takes records
 * from the left-hand branch as input.
 * After retrieving a record from the left-hand branch,
 * it pulls data from the right-hand branch and passes
 * the merged record upwards until the right-hand branch
 * is depleted, at which time data is pulled from the
 * left-hand branch and the process repeats.  */
typedef struct {
	OpBase op;
	Record r;                       // bound branch record
	Record *records;                // LHS records
	OpBase *bound_branch;           // bound branch
	OpBase *rhs_branch;             // right-hand branch
	Argument *op_arg;               // right-hand branch tap
} Apply;

OpBase *NewApplyOp(const ExecutionPlan *plan);
