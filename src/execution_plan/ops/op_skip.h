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
	unsigned int skip;    // number of records to skip
	unsigned int skipped; // number of records already skipped
	AR_ExpNode *skip_exp; // expression evaluated to 'skip'
} OpSkip;

// Skips 'n' records.
OpBase *NewSkipOp(const ExecutionPlan *plan, AR_ExpNode *skip_exp);

