/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../ops/op.h"
#include "../execution_plan.h"
#include "../../util/rax_extensions.h"

// returns true if an operation in the op-tree rooted at `root` is eager
bool ExecutionPlan_isEager
(
    OpBase *root
);

//------------------------------------------------------------------------------
// ExecutionPlan_Locate API:
// For performing existence checks and looking up individual operations in tree.
//------------------------------------------------------------------------------

// locate the first operation matching one of the given types in the op tree by
// performing DFS
// returns NULL if no matching operation was found
OpBase *ExecutionPlan_LocateOpMatchingTypes
(
    OpBase *root,
    const OPType *types,
    uint type_count
);

// Convenience wrapper around ExecutionPlan_LocateOpMatchingType for lookups of a single type.
// Locate the first operation of a given type within execution plan by performing DFS.
// Returns NULL if operation wasn't found
OpBase *ExecutionPlan_LocateOp
(
    OpBase *root,
    OPType type
);

// searches for an operation of a given type, up to the given depth in the
// execution-plan
OpBase *ExecutionPlan_LocateOpDepth
(
    OpBase *root,
    OPType type,
    uint depth
);

// find the earliest operation at which all references are resolved, if any,
// without recursing past a blacklisted op
OpBase *ExecutionPlan_LocateReferencesExcludingOps
(
	OpBase *root,                   // start point
	const OPType *blacklisted_ops,  // blacklisted operations
	int nblacklisted_ops,           // number of blacklisted operations
	rax *refs_to_resolve            // references to resolve
);

// scans plan from root via parent nodes until a limit operation is found
// eager operation will terminate the scan
// return true if a limit operation was found, in which case 'limit' is set
// otherwise return false
bool ExecutionPlan_ContainsLimit
(
	OpBase *root,    // root to start the scan from
	uint64_t *limit  // limit value
);

// scans plan from root via parent nodes until a skip operation is found
// eager operation will terminate the scan
// return true if a skip operation was found, in which case 'skip' is set
// otherwise return false
bool ExecutionPlan_ContainsSkip
(
	OpBase *root,   // root to start the scan from
	uint64_t *skip  // skip value
);

//------------------------------------------------------------------------------
// ExecutionPlan_Collect API:
// For collecting all matching operations in tree.
//------------------------------------------------------------------------------

// Collect all operations matching the given types in the op tree.
// Returns an array of operations
OpBase **ExecutionPlan_CollectOpsMatchingTypes
(
    OpBase *root,
    const OPType *types,
    uint type_count
);

// Convenience wrapper around ExecutionPlan_LocateOpMatchingType for
// collecting all operations of a given type within the op tree.
// Returns an array of operations
OpBase **ExecutionPlan_CollectOps
(
    OpBase *root,
    OPType type
);

// fills `ops` with all operations from `op` an upward (towards parent) in the
// execution plan
// returns the amount of ops collected
uint ExecutionPlan_CollectUpwards
(
    OpBase *ops[],
    OpBase *op
);

//------------------------------------------------------------------------------
// API for building and relocating operations in transient ExecutionPlans
//------------------------------------------------------------------------------

// populate a rax with all aliases that have been resolved by the given operation
// and its children
// these are the bound variables at this point in execution, and
// subsequent operations should not introduce them as new entities
// for example, in the query:
// MATCH (a:A) CREATE (a)-[:E]->(b:B)
// the Create operation should not introduce a new node 'a'
void ExecutionPlan_BoundVariables
(
    const OpBase *op,
    rax *modifiers,
    const ExecutionPlan *plan
);

