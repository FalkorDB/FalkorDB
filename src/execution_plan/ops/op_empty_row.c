/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "op_empty_row.h"

// forward declarations
static Record EmptyRowConsume(OpBase *opBase);
static OpResult EmptyRowReset(OpBase *opBase) ;
static OpBase *EmptyRowClone(const ExecutionPlan *plan, const OpBase *opBase);

OpBase *NewEmptyRow
(
	const ExecutionPlan *plan
) {
	OpEmptyRow *op = rm_calloc (1, sizeof (OpEmptyRow)) ;
	op->emit = true ;

	// set operations
	OpBase_Init ((OpBase *)op, OPType_EMPTY_ROW, "EmptyRow", NULL,
			EmptyRowConsume, EmptyRowReset, NULL, EmptyRowClone, NULL, false,
			plan) ;

	return (OpBase *)op ;
}

static Record EmptyRowConsume
(
	OpBase *opBase
) {
	OpEmptyRow *op = (OpEmptyRow*)opBase ;
	Record r = NULL ;

	if (op->emit) {
		r = OpBase_CreateRecord (opBase) ;
		op->emit = false ;
	}

	return r ;
}

static OpResult EmptyRowReset
(
	OpBase *ctx
) {
	OpEmptyRow *op = (OpEmptyRow *)ctx ;
	op->emit = true ;

	return OP_OK ;
}

static inline OpBase *EmptyRowClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_EMPTY_ROW);

	return NewEmptyRow (plan) ;
}

