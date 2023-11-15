/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "op_empty_record.h"

// forward declarations
static OpResult EmptyRecordInit(OpBase *opBase);
static Record EmptyRecordConsume(OpBase *opBase);
static OpResult EmptyRecordReset(OpBase *opBase);
static OpBase *EmptyRecordClone(const ExecutionPlan *plan, const OpBase *opBase);
static void EmptyRecordFree(OpBase *opBase);

OpBase *NewEmptyRecordOp
(
	const ExecutionPlan *plan
) {
	OpEmptyRecord *op = rm_calloc(1, sizeof(OpEmptyRecord));

	// set Op operations
	OpBase_Init((OpBase *)op, OPType_EMPTY_RECORD, "Empty Record",
			EmptyRecordInit, EmptyRecordConsume, EmptyRecordReset, NULL,
			EmptyRecordClone, EmptyRecordFree, false, plan);

	return (OpBase*)op;
}

static OpResult EmptyRecordInit
(
	OpBase *opBase
) {
	// set op's empty record
	OpEmptyRecord *op = (OpEmptyRecord*)opBase;

	ASSERT(op    != NULL);
	ASSERT(op->r == NULL);
	ASSERT(OpBase_ChildCount(opBase) == 0);  // validate op is a tap

	// get an empty record
	op->r = OpBase_CreateRecord(opBase);
	ASSERT(op->r != NULL);

	return OP_OK;
}

static Record EmptyRecordConsume
(
	OpBase *opBase
) {
	ASSERT(opBase != NULL);

	OpEmptyRecord *op = (OpEmptyRecord*)opBase;

	// emit op's record only once
	Record r = op->r;
	op->r = NULL;
	return r;
}

static OpResult EmptyRecordReset
(
	OpBase *opBase
) {
	ASSERT(opBase != NULL);

	// create a new empty record if the former was emitted
	OpEmptyRecord *op = (OpEmptyRecord*)opBase;
	if(op->r == NULL) op->r = OpBase_CreateRecord(opBase);

	return OP_OK;
}

static OpBase *EmptyRecordClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	return NewEmptyRecordOp(plan);
}

static void EmptyRecordFree
(
	OpBase *opBase
) {
	ASSERT(opBase != NULL);
	OpEmptyRecord *op = (OpEmptyRecord*)opBase;
	
	if(op->r != NULL) {
		OpBase_DeleteRecord(op->r);
		op->r = NULL;
	}
}

