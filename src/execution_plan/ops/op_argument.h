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

/* The Argument operation holds an internal Record that it will emit exactly once. */
typedef struct {
	OpBase op;
	Record r;
} Argument;

OpBase *NewArgumentOp(const ExecutionPlan *plan, const char **variables);

void Argument_AddRecord(Argument *arg, Record r);

