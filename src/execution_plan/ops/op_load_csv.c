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
static Record LoadCSVConsumeDepleted(OpBase *opBase);
static Record LoadCSVConsumeFromChild(OpBase *opBase);
static OpBase *LoadCSVClone(const ExecutionPlan *plan, const OpBase *opBase);
static OpResult LoadCSVReset(OpBase *opBase);
static void LoadCSVFree(OpBase *opBase);

// evaluate path expression
// expression must evaluate to string representing a valid URI
// if that's not the case an exception is raised
static bool _compute_path
(
	OpLoadCSV *op,
	Record r
) {
	ASSERT(op != NULL);

	op->path = AR_EXP_Evaluate(op->exp, r);
	if(SI_TYPE(op->path) != T_STRING) {
		ErrorCtx_RaiseRuntimeException("path to CSV must be a string");
		return false;
	}

	return true;
}

// mock data
static char *A = "AAA";
static char *B = "BB";
static char *C = "C";

// get a single CSV row
static bool _CSV_GetRow
(
	OpLoadCSV *op,  // load CSV operation
	SIValue *row    // row to populate
) {
	ASSERT(op  != NULL);
	ASSERT(row != NULL);

	if(!op->mock) return false;

	SIValue _row = SIArray_New(3);

	SIArray_Append(&_row, SI_ConstStringVal(A));
	SIArray_Append(&_row, SI_ConstStringVal(B));
	SIArray_Append(&_row, SI_ConstStringVal(C));

	*row = _row;

	op->mock = false;
	return true;
}

// create a new load CSV operation
OpBase *NewLoadCSVOp
(
	const ExecutionPlan *plan,  // execution plan
	AR_ExpNode *exp,            // CSV URI path expression
	const char *alias,          // CSV row alias
	bool with_headers           // CSV contains header row
) {
	ASSERT(exp   != NULL);
	ASSERT(plan  != NULL);
	ASSERT(alias != NULL);

	OpLoadCSV *op = rm_calloc(1, sizeof(OpLoadCSV));

	op->exp          = exp;
	op->path         = SI_NullVal();
	op->mock         = true;
	op->alias        = strdup(alias);
	op->with_headers = with_headers;

	// Set our Op operations
	OpBase_Init((OpBase *)op, OPType_LOAD_CSV, "Load CSV", LoadCSVInit,
			LoadCSVConsume, NULL, NULL, LoadCSVClone, LoadCSVFree, false, plan);

	op->recIdx = OpBase_Modifies((OpBase *)op, alias);

	return (OpBase *)op;
}

// initialize operation
static OpResult LoadCSVInit
(
	OpBase *opBase
) {
	// set operation consume function

	OpLoadCSV *op = (OpLoadCSV*)opBase;
	// update consume function in case operation has a child
	if(OpBase_ChildCount(opBase) > 0) {
		op->child = OpBase_GetChild(opBase, 0);
		OpBase_UpdateConsume(opBase, LoadCSVConsumeFromChild);	

		return OP_OK;
	}

	// no child operation evaluate path expression
	Record r = OpBase_CreateRecord(opBase);
	if(!_compute_path(op, r)) {
		// failed to evaluate CSV path
		// update consume function
		OpBase_DeleteRecord(r);
		OpBase_UpdateConsume(opBase, LoadCSVConsumeDepleted);
	}

	return OP_OK;
}

// simply return NULL indicating operation depleted
static Record LoadCSVConsumeDepleted
(
	OpBase *opBase
) {
	return NULL;
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

		// first call, evaluate CSV path
		if(!_compute_path(op, op->child_record)) {
			// failed to evaluate CSV path, quickly return
			return NULL;
		}

		op->mock = true;
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

		// free CSV path, just in case it relies on record data
		SIValue_Free(op->path);
		op->path = SI_NullVal();

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
	return NewLoadCSVOp(plan, AR_EXP_Clone(op->exp), op->alias,
			op->with_headers);
}

static OpResult LoadCSVReset (
	OpBase *opBase
) {
	OpLoadCSV *op = (OpLoadCSV*)opBase;

	SIValue_Free(op->path);
	op->path = SI_NullVal();

	if(op->child_record != NULL) {
		OpBase_DeleteRecord(op->child_record);
		op->child_record = NULL;
	}

	return OP_OK;
}

// free Load CSV operation
static void LoadCSVFree
(
	OpBase *opBase
) {
	ASSERT(opBase != NULL);

	OpLoadCSV *op = (OpLoadCSV*)opBase;

	SIValue_Free(op->path);

	if(op->exp != NULL) {
		AR_EXP_Free(op->exp);
		op->exp = NULL;
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

