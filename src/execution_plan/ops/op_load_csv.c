/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "op_load_csv.h"
#include "../../datatypes/array.h"

// forward declarations
static OpResult LoadCSVInit(OpBase *opBase);
static Record LoadCSVConsume(OpBase *opBase);
static Record LoadCSVConsumeFromChild(OpBase *opBase);
static OpBase *LoadCSVClone(const ExecutionPlan *plan, const OpBase *opBase);
static void LoadCSVFree(OpBase *opBase);

// create a new load CSV operation
OpBase *NewLoadCSVOp
(
	const ExecutionPlan *plan,  // execution plan
	const char *path,           // path to CSV file
	const char *alias           // CSV row alias
) {
	ASSERT(plan  != NULL);
	ASSERT(path  != NULL);
	ASSERT(alias != NULL);

	OpLoadCSV *op = rm_calloc(1, sizeof(OpLoadCSV));

	op->path  = strdup(path);
	op->alias = strdup(alias);

	// Set our Op operations
	OpBase_Init((OpBase *)op, OPType_LOAD_CSV, "Load CSV", LoadCSVInit,
			LoadCSVConsume, NULL, NULL, LoadCSVClone, LoadCSVFree, false, plan);

	op->recIdx = OpBase_Modifies((OpBase *)op, alias);

	return (OpBase *)op;
}

static OpResult LoadCSVInit
(
	OpBase *opBase
) {
	// update consume function in case operation has a child
	if(OpBase_ChildCount(opBase) > 0) {
		OpLoadCSV *op = (OpLoadCSV*)opBase;
		op->child = OpBase_GetChild(opBase, 0);
		OpBase_UpdateConsume(opBase, LoadCSVConsumeFromChild);	
	}

	return OP_OK;
}

static const char *A = "AAA";
static const char *B = "BB";
static const char *C = "C";

static bool _CSV_GetRow
(
	OpLoadCSV *op,
	SIValue *row
) {
	ASSERT(op  != NULL);
	ASSERT(row != NULL);

	SIValue _row = SIArray_New(3);

	SIArray_Append(&_row, SI_ConstStringVal(A));
	SIArray_Append(&_row, SI_ConstStringVal(B));
	SIArray_Append(&_row, SI_ConstStringVal(C));

	*row = _row;

	return true;
}

// load CSV consume function in case this operation in not a tap
static Record LoadCSVConsumeFromChild
(
	OpBase *opBase
) {
	ASSERT(opBase != NULL);

	OpLoadCSV *op = (OpLoadCSV*)opBase;

pull_from_child:
	// in case a record is missing ask child to provide one
	if(op->child_record == NULL) {
		op->child_record = OpBase_Consume(op->child);

		// child failed to provide record, depleted
		if(op->child_record == NULL) {
			return NULL;
		}
	}

	// must have a record at this point
	ASSERT(op->child_record != NULL);

	// get a new CSV row
	SIValue row;
	if(!_CSV_GetRow(op, &row)) {
		// failed to get a CSV row
		// reset CSV reader and free current child record
		OpBase_DeleteRecord(op->child_record);
		op->child_record = NULL;

		// try to get a new record from child
		goto pull_from_child;
	}

	// managed to get a new CSV row
	// update record and return to caller
	Record r = OpBase_DeepCloneRecord(op->child_record);
	Record_AddScalar(r, op->recIdx, row);

	return r;
}

// load CSV consume function in the case this operation is a tap
static Record LoadCSVConsume
(
	OpBase *opBase
) {
	ASSERT(opBase != NULL);

	OpLoadCSV *op = (OpLoadCSV*)opBase;

	SIValue row;
	Record r = NULL;

	if(_CSV_GetRow(op, &row)) {
		r = OpBase_CreateRecord(opBase);
		Record_AddScalar(r, op->recIdx, row);
	}

	return r;
}

static inline OpBase *LoadCSVClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_LOAD_CSV);

	OpLoadCSV *op = (OpLoadCSV*)opBase;
	return NewLoadCSVOp(plan, op->path, op->alias);
}

// free Load CSV operation
static void LoadCSVFree
(
	OpBase *opBase
) {
	ASSERT(opBase != NULL);

	OpLoadCSV *op = (OpLoadCSV*)opBase;

	if(op->path != NULL) {
		rm_free(op->path);
		op->path = NULL;	
	}

	if(op->alias != NULL) {
		rm_free(op->alias);
		op->alias = NULL;
	}
	
	if(op->child_record != NULL) {
		OpBase_DeleteRecord(op->child_record);
		op->child_record = NULL;	
	}
}

