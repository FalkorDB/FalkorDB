/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"

// the OpArgument operation holds a single Record that it will emit once
// acting as the input/tap for a branch of operations
typedef struct {
	OpBase op;
	Record r;   // record to emit
	Record _r;  // copy of 'r' used to restore 'r' incase the operation resets
} OpArgument;

// create a new OpArgument operation
OpBase *NewArgumentOp
(
	const ExecutionPlan *plan,  // execution plan
	const char **variables      // variables introduced by operation
);

// set's the operation record
void Argument_AddRecord
(
	OpArgument *arg,  // argument operation
	Record r          // record to set
);

