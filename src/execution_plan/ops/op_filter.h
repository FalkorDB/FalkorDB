/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"
#include "../../filter_tree/filter_tree.h"

/* Filter
 * filters graph according to where cluase */
typedef struct {
	OpBase op;
	FT_FilterNode *filterTree;
} OpFilter;

/* Creates a new Filter operation */
OpBase *NewFilterOp(const ExecutionPlan *plan, FT_FilterNode *filterTree);
