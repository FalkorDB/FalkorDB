/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../ops/op.h"
typedef struct ExecutionPlan ExecutionPlan;

//------------------------------------------------------------------------------
// helper functions to modify execution plans
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//  API for restructuring the op tree
//------------------------------------------------------------------------------

// remove entire branch
void ExecutionPlan_RemoveBranch
(
	OpBase *root  // root of branch to remove
);

// remove node from its parent
// parent inherits all children of node
void ExecutionPlan_RemoveOp
(
	OpBase *op  // op to remove
);

// push b right below a
void ExecutionPlan_PushBelow
(
	OpBase *a,
	OpBase *b
);

// replace a with b
void ExecutionPlan_ReplaceOp
(
	OpBase *a,
	OpBase *b
);

// update the root op of the execution plan
void ExecutionPlan_UpdateRoot
(
	ExecutionPlan *plan,  // plan to update
	OpBase *new_root      // new root op
);


//------------------------------------------------------------------------------
//  API for binding ops to plans
//------------------------------------------------------------------------------

// for all ops in the given tree, associate the provided ExecutionPlan
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

