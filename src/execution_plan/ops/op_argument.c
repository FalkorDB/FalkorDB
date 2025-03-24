/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_argument.h"
#include "RG.h"

// forward declarations
static Record ArgumentConsume(OpBase *opBase);
static OpResult ArgumentReset(OpBase *opBase);
static OpBase *ArgumentClone(const ExecutionPlan *plan, const OpBase *opBase);
static void ArgumentFree(OpBase *opBase);

// create a new OpArgument operation
OpBase *NewArgumentOp
(
	const ExecutionPlan *plan,  // execution plan
	const char **variables      // variables introduced by operation
) {
	ASSERT(plan != NULL);

	OpArgument *op = rm_calloc(1, sizeof(OpArgument));

	// set our Op operations
	OpBase_Init((OpBase *)op, OPType_ARGUMENT, "Argument", NULL,
				ArgumentConsume, ArgumentReset, NULL, ArgumentClone,
				ArgumentFree, false, plan);

	// introduce modifies
	uint n = array_len(variables);
	for(uint i = 0; i < n; i++) {
		OpBase_Modifies((OpBase *)op, variables[i]);
	}

	return (OpBase *)op;
}

// consume function
// emit internal record once
static Record ArgumentConsume
(
	OpBase *opBase
) {
	ASSERT(opBase != NULL);

	OpArgument *arg = (OpArgument *)opBase;

	// emit the record only once
	// arg->r can already be NULL if the op is depleted
	Record r = arg->r;

	arg->r = NULL;
	return r;
}

// reset function
// restore 'r'
static OpResult ArgumentReset
(
	OpBase *opBase
) {
	ASSERT(opBase != NULL);

	OpArgument *arg = (OpArgument *)opBase;

	// restore original record if 'r' was emitted
	// and '_r' is present
	if(arg->r == NULL && arg->_r != NULL) {
		// restore original record
		arg->r = arg->_r;

		// create a copy for later restoration
		arg->_r = OpBase_CreateRecord(opBase);
		Record_Clone(arg->r, arg->_r);
	}

	return OP_OK;
}


// set's the operation record
void Argument_AddRecord
(
	OpArgument *arg,  // argument operation
	Record r          // record to set
) {
	ASSERT(r   != NULL);
	ASSERT(arg != NULL);

	// free current 'r' and '_r'
	if(arg->r != NULL) {
		OpBase_DeleteRecord(&arg->r);
	}

	if(arg->_r != NULL) {
		OpBase_DeleteRecord(&arg->_r);
	}

	// set record
	arg->r = r;

	// backup record for later restoration upon reset
	arg->_r = OpBase_CreateRecord((OpBase*)arg);
	Record_Clone(r, arg->_r);
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
	OpArgument *arg = (OpArgument *)opBase;

	// free record and its backup

	if(arg->r != NULL) {
		OpBase_DeleteRecord(&arg->r);
		arg->r = NULL;
	}

	if(arg->_r != NULL) {
		OpBase_DeleteRecord(&arg->_r);
		arg->_r = NULL;
	}
}

