/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "op_argument.h"
#include "../execution_plan.h"

// semiApply operation tests for the presence of a pattern
// normal Semi Apply: starts by pulling on the main execution plan branch
// for each record received it tries to get a record from the match branch
// if no data is produced it will try to fetch a new data point from the main
// execution plan branch otherwise the main execution plan branch record is
// passed onward
// Anti Semi Apply: starts by pulling on the main execution plan branch
// for each record received it tries to get a record from the match branch
// if no data is produced the main execution plan branch record is passed onward
// otherwise it will try to fetch a new data point from the main branch

typedef struct OpSemiApply {
	OpBase op;
	Record r;              // bound branch record
	OpBase *bound_branch;  // bound branch root
	OpBase *match_branch;  // match branch root
} OpSemiApply;

OpBase *NewSemiApplyOp
(
	const ExecutionPlan *plan,
	bool anti
);

