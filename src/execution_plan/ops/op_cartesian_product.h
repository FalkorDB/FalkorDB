/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"

// Cartesian product AKA Join
typedef struct {
	OpBase op;
	bool init;
	Record *records;
} CartesianProduct;

OpBase *NewCartesianProductOp
(
	const ExecutionPlan *plan
);
