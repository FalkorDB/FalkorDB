/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"

// cartesian product
typedef struct {
	OpBase op;
	Record r;
	bool init;
} CartesianProduct;

OpBase *NewCartesianProductOp
(
	const ExecutionPlan *plan
);

