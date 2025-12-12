/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "../ops/op.h"
#include "../ops/op_conditional_traverse.h"
#include "../execution_plan_build/execution_plan_util.h"
#include "../execution_plan_build/execution_plan_modify.h"

static void _reduceOptionalMatch
(
	ExecutionPlan *plan,
	OpBase *apply
) {
	OPType t ;
	OpBase *rhs ;

	//--------------------------------------------------------------------------
	// match pattern
	//--------------------------------------------------------------------------

	rhs = OpBase_GetChild (apply, 1) ;
	t = OpBase_Type (rhs) ;
	if (t != OPType_OPTIONAL) {
		return ;
	}

	// Ensure Optional has at least one child before attempting to access it
	if (rhs->childCount == 0) {
		return ;
	}

	rhs = OpBase_GetChild (rhs, 0) ;
	t = OpBase_Type (rhs) ;
	if (t != OPType_CONDITIONAL_TRAVERSE) {
		// Skip optimization if the RHS doesn't start with ConditionalTraverse.
		// This can happen when the OPTIONAL MATCH clause contains multiple
		// comma-separated patterns (e.g., "p0 = (n1), p1 = (n0)"), which may
		// produce a CartesianProduct or other non-ConditionalTraverse subtree.
		// The optimization assumes a strict pattern:
		// Apply → Optional → ConditionalTraverse* → Argument
		// and is not safe to apply to other tree shapes.
		return ;
	}

	// Traverse down the chain of ConditionalTraverse operations,
	// ensuring we only have ConditionalTraverse nodes until we reach Argument
	while (t == OPType_CONDITIONAL_TRAVERSE) {
		// Ensure the ConditionalTraverse has a child before accessing it
		if (rhs->childCount == 0) {
			return ;
		}
		rhs = OpBase_GetChild (rhs, 0) ;
		t = OpBase_Type (rhs) ;
	}

	if (t != OPType_ARGUMENT) {
		// Skip optimization if we encounter any operation type other than
		// ConditionalTraverse before reaching the Argument.
		// This ensures we only optimize the supported pattern.
		return ;
	}

	// pattern matched!
	// perform the following modifications:
	// 1. remove optional
	// 2. make every conditional traverse optional
	// 3. remove argument
	// 4. remove apply

	OpBase *optional = OpBase_GetChild (apply, 1) ;
	OpBase *op       = OpBase_GetChild (optional, 0) ;

	//--------------------------------------------------------------------------
	// remove optional
	//--------------------------------------------------------------------------

	ExecutionPlan_RemoveOp (plan, optional) ;
	OpBase_Free (optional) ;

	//--------------------------------------------------------------------------
	// make every conditional traverse optional
	//--------------------------------------------------------------------------

	while (OpBase_Type (op) == OPType_CONDITIONAL_TRAVERSE) {
		CondTraverse_MakeOptional ((OpCondTraverse*)op) ;
		op = OpBase_GetChild (op, 0) ;
	}

	//--------------------------------------------------------------------------
	// remove argument
	//--------------------------------------------------------------------------

	ASSERT (OpBase_Type (op) == OPType_ARGUMENT) ;
	OpBase *head = OpBase_Parent (op) ;
	ExecutionPlan_RemoveOp (plan, op) ;
	OpBase_Free (op) ;

	//--------------------------------------------------------------------------
	// remove apply
	//--------------------------------------------------------------------------

	OpBase *lhs = OpBase_GetChild (apply, 0) ;
	ExecutionPlan_DetachOp (lhs) ;
	ExecutionPlan_AddOp (head, lhs) ;

	ExecutionPlan_RemoveOp (plan, apply) ;
	OpBase_Free (apply) ;
}

// simplifies Apply + Optional + Traverse + Argument
// patterns into:
// Optional Conditional Traverse
//
// by doing so we transition from a single source BFS into multi-source
// batch traversal
//
// e.g.
//
// Apply
//      Node By Label Scan
//      Optional
//          Conditional Traverse
//              Argument
//
// ->
//
// Optional Conditional Traverse
//     Node By Label Scan
//
//------------------------------------------------------------------------------
//
// Apply
//     Optional
//         Conditional Traverse
//             Node By Label Scan
//      Optional
//          Conditional Traverse
//              Argument
//
// ->
//
// Optional Conditional Traverse
//     Optional
//         Conditional Traverse
//             Node By Label Scan
//      
void batchOptionalMatch
(
	ExecutionPlan *plan
) {
	ASSERT (plan != NULL) ;

	// search for the pattern:
	//
	// Apply
	//      LHS
	//      Optional
	//          Conditional Traverse
	//              Argument


	OpBase **applies = ExecutionPlan_CollectOps (plan->root, OPType_APPLY) ;
	
	uint n = array_len (applies) ;
	for (uint i = 0; i < n; i++) {
		_reduceOptionalMatch (plan, applies[i]) ;
	}

	array_free (applies) ;
}

