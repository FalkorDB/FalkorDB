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

// the ArgumentList operation holds an internal Record list that it emits
// one-by-one exactly once

typedef struct {
	OpBase op;        // OpBase
	Record *records;  // record list
	uint rec_len;     // record list length
} ArgumentList;

OpBase *NewArgumentListOp
(
	const ExecutionPlan *plan
);

void ArgumentList_AddRecordList
(
	ArgumentList *arg,  // Argument operation to plant the list in
	Record *records     // record list
);

