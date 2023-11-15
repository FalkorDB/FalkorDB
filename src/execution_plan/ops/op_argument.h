/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"

// argument operation holds an internal Record that it will emit exactly once
typedef struct {
	OpBase op;
	OpBase *producer;     // operation providing us with data
} Argument;

OpBase *NewArgumentOp
(
	const ExecutionPlan *plan,
	const char **variables
);

