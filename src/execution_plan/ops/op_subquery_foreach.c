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
static Record SubQueryForeachConsumeEager(OpBase *opBase);
static OpResult SubQueryForeachReset(OpBase *opBase);
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
			SubQueryForeachInit, SubQueryForeachConsume, SubQueryForeachReset,
			NULL, SubQueryForeachClone, SubQueryForeachFree, false, plan) ;

	return (OpBase *)op ;
}

static OpResult SubQueryForeachInit
(
	OpBase *opBase
) {
	OpSubQueryForeach *op = (OpSubQueryForeach*)opBase;

	// determine if sub-query is eager or not
	ASSERT (OpBase_ChildCount (opBase) == 2) ;  // expecting 2 children

	OpBase *sub_query_root = OpBase_GetChild (opBase, 1) ;
	op->eager = ExecutionPlan_isEager (sub_query_root) ;

	if (op->eager) {
		OpBase_UpdateConsume (opBase, SubQueryForeachConsumeEager) ;
		op->records = array_new (Record, 1) ;
	}

	//--------------------------------------------------------------------------
	// search for taps
	//--------------------------------------------------------------------------

	op->taps     = array_new (OpBase*, 1) ;
	OpBase **ops = array_new (OpBase*, 1) ;

	array_append (ops, sub_query_root) ;

	while (array_len (ops) > 0) {
		OpBase *child = array_pop (ops) ;
		OPType t = OpBase_Type (child) ;

		// tap located
		if ((OpBase_ChildCount (child) == 0) &&
			(t == OPType_ARGUMENT || t == OPType_ARGUMENT_LIST)) {
			array_append (op->taps, child) ;
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

	// make sure all taps are of the same type
	if (op->n_taps > 0) {
		OPType t = OpBase_Type (op->taps[0]) ;
		for (uint i = 1; i < op->n_taps; i++) {
			ASSERT (OpBase_Type (op->taps[i]) == t) ;
		}

		// argument for non eager, argument-list for eager
		ASSERT ((t == OPType_ARGUMENT && !op->eager) ||
				(t == OPType_ARGUMENT_LIST && op->eager)) ;
	}

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

		OpArgument *tap = (OpArgument*) op->taps[i] ;
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

	// return record
	return r ;
}

static Record SubQueryForeachConsumeEager
(
	OpBase *opBase
) {
	OpSubQueryForeach *op = (OpSubQueryForeach*)opBase;

handoff:
	if (array_len (op->records) > 0) {
		return array_pop (op->records) ;
	}

	OpBase *input = OpBase_GetChild (opBase, 0) ;

	// drain input stream
	Record r ;
	while ((r = OpBase_Consume (input))) {
		array_append (op->records, r) ;
	}

	// see if we've consumed any records
	if (array_len (op->records) == 0) {
		return NULL ;
	}

	//--------------------------------------------------------------------------
	// plant records in sub-query taps
	//--------------------------------------------------------------------------

	for (uint i = 0; i < op->n_taps; i++) {
		// clone records for each sub-query stream
		uint n_rec = array_len (op->records) ;
		Record *clone_records = array_new (Record, n_rec) ;
		for (uint j = 0; j < n_rec; j++) {
			Record clone = OpBase_CreateRecord (opBase) ;
			Record_Clone (op->records[j], clone) ;
			array_append (clone_records, clone) ;
		}

		ArgumentList *tap = (ArgumentList*) op->taps[i] ;
		ArgumentList_AddRecordList (tap, clone_records) ;
	}

	//--------------------------------------------------------------------------
	// drain sub-query
	//--------------------------------------------------------------------------

	OpBase *sub_query_root = OpBase_GetChild (opBase, 1) ;
	Record ignore ;
	while ((ignore = OpBase_Consume (sub_query_root))) {
		OpBase_DeleteRecord (&ignore) ;
	}

	goto handoff;
}

static OpResult SubQueryForeachReset
(
	OpBase *OpBase
) {
	OpSubQueryForeach *op = (OpSubQueryForeach*)OpBase ;

	if (op->records != NULL) {
		uint n = array_len (op->records) ;
		for (uint i = 0; i < n; i++) {
			OpBase_DeleteRecord (op->records + i) ;
		}
		array_clear (op->records) ;
	}

	return OP_OK ;
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

	if (op->records != NULL) {
		uint n = array_len (op->records) ;
		for (uint i = 0; i < n; i++) {
			OpBase_DeleteRecord (op->records + i) ;
		}
		op->records = NULL ;
	}
}

