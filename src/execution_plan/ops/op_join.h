/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
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
