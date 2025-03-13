/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../ops/op.h"

// update execution plan awareness due to the addition of an operation
// when an operation is added by a call to 'ExecutionPlan_AddOp'
// we need to add each of the op's modifiers (aliases introduced by the op)
// to each parent operation
void ExecutionPlanAwareness_AddOp
(
	const OpBase *op
);

// update execution plan awareness due to the removal of an operation
// when an operation is removed by a call to 'ExecutionPlan_RemoveOp'
// we need to remove each of the op's modifiers (aliases introduced by the op)
// from each parent operation
void ExecutionPlanAwareness_RemoveOp
(
	const OpBase *op  // removed op
);

// update execution plan awareness due to the removal of an entire branch
// rooted at op
// when a branch is removed by a call to 'ExecutionPlan_DetachOp'
// we need to remove all variables the detached branch is aware of
// from each parent operation
void ExecutionPlanAwareness_RemoveBranch
(
	const OpBase *root  // branch root
);

