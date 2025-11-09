/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_optional.h"

/* Forward declarations. */
static Record OptionalConsume(OpBase *opBase);
static OpResult OptionalReset(OpBase *opBase);
static OpBase *OptionalClone(const ExecutionPlan *plan, const OpBase *opBase);

OpBase *NewOptionalOp(const ExecutionPlan *plan) {
	Optional *op = rm_calloc (1, sizeof(Optional)) ;

	// Set our Op operations
	OpBase_Init((OpBase *)op, OPType_OPTIONAL, "Optional", NULL, OptionalConsume,
				OptionalReset, NULL, OptionalClone, NULL, false, plan);

	return (OpBase *)op;
}

static Record _DepleteConsume(OpBase *opBase) {
	return NULL;
}

static Record OptionalConsume(OpBase *opBase) {
	Optional *op = (Optional *)opBase;
	// try to produce a Record from the child op.
	Record r = OpBase_Consume(opBase->children[0]);

	// create an empty Record if the child returned NULL
	// and this op has not yet returned data.
	if(!r && !op->emitted_record) {
		r = OpBase_CreateRecord(opBase);
		OpBase_UpdateConsume(opBase, _DepleteConsume);
	}

	// don't produce multiple empty Records.
	op->emitted_record = true;

	return r;
}

static OpResult OptionalReset(OpBase *opBase) {
	Optional *op = (Optional *)opBase;
	op->emitted_record = false;
	OpBase_UpdateConsume(opBase, OptionalConsume);
	return OP_OK;
}

static inline OpBase *OptionalClone(const ExecutionPlan *plan, const OpBase *opBase) {
	Optional *op = (Optional *)opBase;
	OpBase *clone = NewOptionalOp(plan);
	return clone;
}
