/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_unwind.h"
#include "../../query_ctx.h"
#include "../../errors/errors.h"
#include "../../datatypes/array.h"
#include "../../arithmetic/arithmetic_expression.h"
#include "limits.h"

// forward declarations
static void UnwindFree(OpBase *opBase);
static OpResult UnwindInit(OpBase *opBase);
static Record UnwindConsume(OpBase *opBase);
static OpResult UnwindReset(OpBase *opBase);
static OpBase *UnwindClone(const ExecutionPlan *plan, const OpBase *opBase);

OpBase *NewUnwindOp
(
	const ExecutionPlan *plan,
	AR_ExpNode *exp
) {
	OpUnwind *op = rm_malloc(sizeof(OpUnwind));

	op->exp           = exp;
	op->list          = SI_NullVal();
	op->listIdx       = 0;
	op->listLen       = 0;
	op->currentRecord = NULL;

	// Set our Op operations
	OpBase_Init((OpBase *)op, OPType_UNWIND, "Unwind", UnwindInit, UnwindConsume,
				UnwindReset, NULL, UnwindClone, UnwindFree, false, plan);

	op->unwindRecIdx = OpBase_Modifies((OpBase *)op, exp->resolved_name);
	return (OpBase *)op;
}

// evaluate list expression,
// if expression did not return a list type value
// creates a list with that value
static void _initList
(
	OpUnwind *op
) {
	// free previous list
	SIValue_Free(op->list);

	// Null-set the list value to avoid memory errors if evaluation fails
	op->list = SI_NullVal(); 
	SIValue new_list = AR_EXP_Evaluate(op->exp, op->currentRecord);
	if(SI_TYPE(new_list) == T_ARRAY) {
		// update the list value.
		op->list = new_list;
	} else if(SI_TYPE(new_list) == T_NULL) {
		op->list = SI_Array(0);
	} else {
		// create a list of size 1 and initialize it with the input exp value
		op->list = SI_Array(1);
		SIArray_Append(&op->list, new_list);
		SIValue_Free(new_list);
	}

	// reset operation list index
	op->listIdx = 0;
	op->listLen = SIArray_Length(op->list);
}

static OpResult UnwindInit
(
	OpBase *opBase
) {
	OpUnwind *op = (OpUnwind *) opBase;
	op->currentRecord = OpBase_CreateRecord((OpBase *)op);

	if(op->op.childCount == 0) {
		// no child operation, list must be static
		_initList(op);
	}

	return OP_OK;
}

// try to generate a new value to return
// NULL will be returned if dynamic list is not evaluated
// or in case where the current list is fully consumed
static Record _handoff
(
	OpUnwind *op
) {
	// if there is a new value ready, return it
	if(op->listIdx < op->listLen) {
		Record  r = OpBase_CloneRecord(op->currentRecord);
		SIValue v = SIArray_Get(op->list, op->listIdx);

		if(!(SI_TYPE(v) & SI_GRAPHENTITY)) {
			SIValue_Persist(&v);
		}

		Record_Add(r, op->unwindRecIdx, v);

		op->listIdx++;
		return r;
	}

	// depleted
	return NULL;
}

static Record UnwindConsume
(
	OpBase *opBase
) {
	OpUnwind *op = (OpUnwind *)opBase;

	// try to produce data
	Record r = _handoff(op);
	if(r != NULL) {
		return r;
	}

	// no child operation to pull data from, we're done
	if(op->op.childCount == 0) {
		return NULL;
	}

	OpBase *child = op->op.children[0];
	// did we manage to get new data?
pull:
	if((r = OpBase_Consume(child))) {
		// free current record
		OpBase_DeleteRecord(&op->currentRecord);

		// assign new record
		op->currentRecord = r;

		// reset index and set list
		_initList(op);

		// skip empty lists
		if(op->listLen == 0) {
			goto pull;
		}
	}

	return _handoff(op);
}

static OpResult UnwindReset
(
	OpBase *ctx
) {
	OpUnwind *op = (OpUnwind *)ctx;

	op->listIdx = 0;
	SIValue_Free(op->list);
	op->list = SI_NullVal();
	op->listLen = 0;

	return OP_OK;
}

static inline OpBase *UnwindClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_UNWIND);

	OpUnwind *op = (OpUnwind *)opBase;
	return NewUnwindOp(plan, AR_EXP_Clone(op->exp));
}

static void UnwindFree
(
	OpBase *ctx
) {
	OpUnwind *op = (OpUnwind *)ctx;
	SIValue_Free(op->list);
	op->list = SI_NullVal();

	if(op->exp) {
		AR_EXP_Free(op->exp);
		op->exp = NULL;
	}

	if(op->currentRecord != NULL) {
		OpBase_DeleteRecord(&op->currentRecord);
	}

	op->currentRecord = NULL;
}

