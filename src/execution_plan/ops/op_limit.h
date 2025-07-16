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
	unsigned int limit;     // Max number of records to consume.
	AR_ExpNode *limit_exp;  // Expression evaluated to limit.
	unsigned int consumed;  // Number of records consumed so far.
} OpLimit;

// Limits number of produced records
OpBase *NewLimitOp(const ExecutionPlan *plan, AR_ExpNode *limit_exp);

