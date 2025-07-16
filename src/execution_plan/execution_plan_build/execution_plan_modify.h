/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "../ops/op.h"
typedef struct ExecutionPlan ExecutionPlan;

//------------------------------------------------------------------------------
// Helper functions to modify execution plans.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//  API for restructuring the op tree.
//------------------------------------------------------------------------------

/* Removes operation from execution plan. */
void ExecutionPlan_RemoveOp(ExecutionPlan *plan, OpBase *op);

/* Detaches operation from its parent. */
void ExecutionPlan_DetachOp(OpBase *op);

/* Adds operation to execution plan as a child of parent. */
void ExecutionPlan_AddOp(OpBase *parent, OpBase *newOp);

// adds child to be the ind'th child of parent
void ExecutionPlan_AddOpInd
(
	OpBase *parent,  // parent op
	OpBase *child,   // child op
	uint ind         // index of child
);

/* Push b right below a. */
void ExecutionPlan_PushBelow(OpBase *a, OpBase *b);

/* Introduce new_root as the parent of old_root. */
void ExecutionPlan_NewRoot(OpBase *old_root, OpBase *new_root);

/* Update the root op of the execution plan. */
void ExecutionPlan_UpdateRoot(ExecutionPlan *plan, OpBase *new_root);

/* Replace a with b. */
void ExecutionPlan_ReplaceOp(ExecutionPlan *plan, OpBase *a, OpBase *b);

//------------------------------------------------------------------------------
//  API for binding ops to plans.
//------------------------------------------------------------------------------

// For all ops in the given tree, associate the provided ExecutionPlan.
// if qg is set, merge the query graphs of the temporary and main plans
void ExecutionPlan_BindOpsToPlan
(
	ExecutionPlan *plan,  // plan to bind the operations to
	OpBase *root,         // root operation
	bool qg               // whether to merge QueryGraphs or not
);

// binds all ops in `ops` to `plan`, except for ops of type `exclude_type`
void ExecutionPlan_MigrateOpsExcludeType
(
	OpBase * ops[],             // array of ops to bind
	OPType exclude_type,        // type of ops to exclude
	uint op_count,              // number of ops in the array
	const ExecutionPlan *plan   // plan to bind the ops to
);
