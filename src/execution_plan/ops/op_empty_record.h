/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"

// the EmptyRecordOperation generates a single empty record
// additional calls to consume on this operation will return NULL
// 
// this operation greatly simplifies the logic of some operations e.g. MERGE
// by normalizing the plan's structure
typedef struct {
	OpBase op;  // base operation
	Record r;   // empty record
} OpEmptyRecord;

OpBase *NewEmptyRecordOp
(
	const ExecutionPlan *plan
);

