/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"

typedef struct {
	OpBase op;
	bool emit;
} OpEmptyRow;

OpBase *NewEmptyRow
(
	const ExecutionPlan *plan
);

