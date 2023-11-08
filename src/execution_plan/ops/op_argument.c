/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "op_argument.h"

// forward declarations
static Record ArgumentConsume(OpBase *opBase);
static OpResult ArgumentReset(OpBase *opBase);
static OpBase *ArgumentClone(const ExecutionPlan *plan, const OpBase *opBase);
static void ArgumentFree(OpBase *opBase);

OpBase *NewArgumentOp
(
	const ExecutionPlan *plan,
	const char **variables
) {
	Argument *op = rm_calloc(1, sizeof(Argument));

	// set our Op operations
	OpBase_Init((OpBase *)op, OPType_ARGUMENT, "Argument", NULL,
			ArgumentConsume, ArgumentReset, NULL, ArgumentClone, ArgumentFree,
			false, plan);

	uint variable_count = array_len(variables);
	for(uint i = 0; i < variable_count; i ++) {
		OpBase_Modifies((OpBase *)op, variables[i]);
	}

	return (OpBase *)op;
}

static Record ArgumentConsume
(
	OpBase *opBase
) {
	Argument *arg = (Argument *)opBase;

	// emit the record only once
	// arg->r can already be NULL if the op is depleted
	Record r = arg->r;
	if(r != NULL) {
		// save emitted record in case we're resetted
		arg->reset_record = OpBase_CreateRecord((OpBase *)arg);
		Record_Clone(r, arg->reset_record);
	}

	arg->r = NULL;
	return r;
}

void Argument_AddRecord
(
	Argument *arg,
	Record r
) {
	ASSERT(r   != NULL);
	ASSERT(arg != NULL);

	// free old record
	if(arg->r != NULL)            OpBase_DeleteRecord(arg->r);
	if(arg->reset_record != NULL) OpBase_DeleteRecord(arg->reset_record);

	arg->r = r;
}

static OpResult ArgumentReset
(
	OpBase *opBase
) {
	// reset operation, freeing the Record if one is held
	Argument *arg = (Argument *)opBase;

	// clear current record
	if(arg->r) {
		OpBase_DeleteRecord(arg->r);
		arg->r = NULL;
	}

	// restore original record
	if(arg->reset_record != NULL) {
		arg->r = arg->reset_record;
		arg->reset_record = NULL;
	}

	return OP_OK;
}

static inline OpBase *ArgumentClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_ARGUMENT);
	return NewArgumentOp(plan, opBase->modifies);
}

static void ArgumentFree
(
	OpBase *opBase
) {
	Argument *arg = (Argument *)opBase;

	if(arg->r) {
		OpBase_DeleteRecord(arg->r);
		arg->r = NULL;
	}

	if(arg->reset_record) {
		OpBase_DeleteRecord(arg->reset_record);
		arg->reset_record = NULL;
	}
}

