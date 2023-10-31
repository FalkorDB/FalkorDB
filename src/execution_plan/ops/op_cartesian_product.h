/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"

// cartesian product AKA join
typedef struct {
	OpBase op;
	Record r;
	bool init;
} CartesianProduct;

OpBase *NewCartesianProductOp
(
	ExecutionPlan *plan
);

