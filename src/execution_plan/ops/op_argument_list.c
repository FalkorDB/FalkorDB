/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op.h"
#include "op_argument_list.h"

// forward declarations
static void ArgumentListFree(OpBase *opBase);
static Record ArgumentListConsume(OpBase *opBase);
static OpResult ArgumentListReset(OpBase *opBase);
static OpBase *ArgumentListClone(const ExecutionPlan *plan, const OpBase *opBase);

OpBase *NewArgumentListOp
(
	const ExecutionPlan *plan,
	const char **variables
) {
	ArgumentList *op = rm_calloc (1, sizeof(ArgumentList)) ;

	// set our Op operations
	OpBase_Init((OpBase *)op, OPType_ARGUMENT_LIST, "Argument List", NULL,
			ArgumentListConsume, ArgumentListReset, NULL, ArgumentListClone,
			ArgumentListFree, false, plan);

	uint variable_count = array_len(variables);
	for(uint i = 0; i < variable_count; i ++) {
		OpBase_Modifies((OpBase *)op, variables[i]);
	}

	return (OpBase *)op;
}

void ArgumentList_AddRecordList
(
	ArgumentList *op,
	Record *records
) {
	ASSERT(op->records == NULL && "insert into a populated ArgumentList");
	op->records = records;
	op->rec_len = array_len(op->records);
}

static Record ArgumentListConsume
(
	OpBase *opBase
) {
	ArgumentList *op = (ArgumentList *)opBase;

	ASSERT(op->records != NULL);

	if(op->rec_len > 0) {
		op->rec_len--;
		return array_pop(op->records);
	}

	return NULL;
}

static OpResult ArgumentListReset
(
	OpBase *opBase
) {
	ArgumentList *op = (ArgumentList *)opBase;

	// free remaining records
	if(op->records != NULL) {
		for(uint i = 0; i < op->rec_len; i++) {
			OpBase_DeleteRecord(op->records+i);
		}

		array_free(op->records);
		op->records = NULL;
	}

	op->rec_len = 0;

	return OP_OK;
}

static inline OpBase *ArgumentListClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_ARGUMENT_LIST);
	return NewArgumentListOp(plan, opBase->modifies);
}

static void ArgumentListFree
(
	OpBase *opBase
) {
	ArgumentList *op = (ArgumentList *)opBase;

	// free remaining records
	if(op->records != NULL) {
		for(uint i = 0; i < op->rec_len; i++) {
			OpBase_DeleteRecord(op->records+i);
		}

		array_free(op->records);
		op->records = NULL;
	}
}
