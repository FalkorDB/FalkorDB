/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "op_argument.h"
#include "op_argument_list.h"
#include "op_subquery_foreach.h"
#include "../execution_plan_build/execution_plan_util.h"

// forward declarations
static OpResult SubQueryForeachInit(OpBase *opBase);
static Record SubQueryForeachConsume(OpBase *opBase);
static void SubQueryForeachFree(OpBase *opBase);
static OpBase *SubQueryForeachClone(const ExecutionPlan *plan, const OpBase *opBase);

OpBase *NewSubQueryForeach
(
	const ExecutionPlan *plan
) {
	// validate inputs
	ASSERT (plan != NULL) ;

	OpSubQueryForeach *op = rm_calloc (1, sizeof (OpSubQueryForeach)) ;

	// set operations
	OpBase_Init ((OpBase *)op, OPType_SUBQUERY_FOREACH, "SubqueryForeach",
			SubQueryForeachInit, SubQueryForeachConsume, NULL, NULL,
			SubQueryForeachClone, SubQueryForeachFree, false, plan) ;

	return (OpBase *)op ;
}

static OpResult SubQueryForeachInit
(
	OpBase *opBase
) {
	OpSubQueryForeach *op = (OpSubQueryForeach*)opBase;

	ASSERT (OpBase_ChildCount (opBase) == 2) ;  // expecting 2 children

	//--------------------------------------------------------------------------
	// search for taps
	//--------------------------------------------------------------------------

	op->taps     = array_new (OpArgument*, 1) ;
	OpBase **ops = array_new (OpBase*, 1) ;

	OpBase *sub_query_root = OpBase_GetChild (opBase, 1) ;
	array_append (ops, sub_query_root) ;

	while (array_len (ops) > 0) {
		OpBase *child = array_pop (ops) ;
		OPType t = OpBase_Type (child) ;

		// tap located
		if ((OpBase_ChildCount (child) == 0) && (t == OPType_ARGUMENT)) {
			array_append (op->taps, (OpArgument*) child) ;
		}

		// join op, traverse each branch
		else if (t == OPType_JOIN || t == OPType_CARTESIAN_PRODUCT) {
			for (uint i = 0; i < OpBase_ChildCount (child); i++) {
				array_append (ops, OpBase_GetChild (child, i)) ;
			}
		}

		// go "left"
		else if (OpBase_ChildCount (child) > 0) {
			array_append (ops, OpBase_GetChild (child, 0)) ;
		}
	}

	op->n_taps = array_len (op->taps) ;

	array_free (ops) ;

	return OP_OK ;
}

static Record SubQueryForeachConsume
(
	OpBase *opBase
) {
	OpSubQueryForeach *op = (OpSubQueryForeach*)opBase;

	OpBase *input = OpBase_GetChild (opBase, 0) ;
	Record r = OpBase_Consume (input) ;
	if (r == NULL) {
		return NULL ;
	}

	//--------------------------------------------------------------------------
	// plant records in sub-query taps
	//--------------------------------------------------------------------------

	for (uint i = 0; i < op->n_taps; i++) {
		Record clone = OpBase_CreateRecord (opBase) ;
		Record_Clone (r, clone) ;

		OpArgument *tap = op->taps[i] ;
		Argument_AddRecord (tap, clone) ;
	}

	//--------------------------------------------------------------------------
	// drain sub-query
	//--------------------------------------------------------------------------

	OpBase *sub_query_root = OpBase_GetChild (opBase, 1) ;
	Record ignore ;
	while ((ignore = OpBase_Consume (sub_query_root))) {
		OpBase_DeleteRecord (&ignore) ;
	}

	// reset sub-query
	OpBase_PropagateReset (sub_query_root) ;

	// return record
	return r ;
}

static inline OpBase *SubQueryForeachClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	return NewSubQueryForeach (plan) ;
}

static void SubQueryForeachFree
(
	OpBase *opBase
) {
	OpSubQueryForeach *op = (OpSubQueryForeach*)opBase;

	if (op->taps != NULL) {
		array_free (op->taps) ;
		op->taps = NULL ;
	}
}

