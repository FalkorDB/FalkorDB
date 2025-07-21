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
