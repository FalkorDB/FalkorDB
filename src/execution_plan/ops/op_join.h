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

typedef struct {
	OpBase op;
	OpBase *stream;          // Current stream to pull from.
	int streamIdx;           // Current stream index.
	bool update_column_map;  // Update column map.
} OpJoin;

OpBase *NewJoinOp(const ExecutionPlan *plan);

bool JoinGetUpdateColumnMap(const OpBase *op);
