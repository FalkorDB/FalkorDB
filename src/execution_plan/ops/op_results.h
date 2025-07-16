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
#include "../../redismodule.h"
#include "../../graph/graphcontext.h"
#include "../../resultset/resultset.h"

/* Results generates result set */

typedef struct {
	OpBase op;
	ResultSet *result_set;
	uint64_t result_set_size_limit;
} Results;

/* Creates a new Results operation */
OpBase *NewResultsOp(const ExecutionPlan *plan);
