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
	unsigned int limit;     // max number of records to consume
	AR_ExpNode *limit_exp;  // expression evaluated to limit
	unsigned int consumed;  // number of records consumed so far
} OpLimit;

// limits number of produced records
OpBase *NewLimitOp
(
	ExecutionPlan *plan,
	AR_ExpNode *limit_exp
);

