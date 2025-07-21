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
#include "rax.h"
#include "../execution_plan.h"

typedef struct {
	OpBase op;
	rax *found;
	rax *mapping;          // record mapping
	uint *offsets;         // offsets to expression values
	const char **aliases;  // expression aliases to distinct by
	uint offset_count;     // number of offsets

} OpDistinct;

OpBase *NewDistinctOp(const ExecutionPlan *plan, const char **aliases, uint alias_count);

