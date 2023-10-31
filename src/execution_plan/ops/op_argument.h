/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"

// argument operation holds an internal Record that it will emit only once
typedef struct {
	OpBase op;
	Record r;
} Argument;

OpBase *NewArgumentOp
(
	ExecutionPlan *plan,
	const char **variables
);

void Argument_AddRecord
(
	Argument *arg,
	Record r
);

