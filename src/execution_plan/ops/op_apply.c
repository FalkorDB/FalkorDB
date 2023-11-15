/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "op_apply.h"
#include "../execution_plan_build/execution_plan_util.h"

// forward declarations
static OpResult ApplyInit(OpBase *opBase);
static Record ApplyConsume(OpBase *opBase);
static OpResult ApplyReset(OpBase *opBase);
static OpBase *ApplyClone(const ExecutionPlan *plan, const OpBase *opBase);
static void ApplyFree(OpBase *opBase);

Record Apply_PullArgRecord
(
	OpBase *op // apply operation (Apply/OpApplyMultiplexer)
) {
	Apply *apply = (Apply*)op;

	ASSERT(apply    != NULL);
	ASSERT(apply->r != NULL);

	Record r = OpBase_CreateRecord(op);
	Record_Clone(apply->r, r);
	return r;
}

OpBase *NewApplyOp
(
	const ExecutionPlan *plan
) {
	Apply *op = rm_calloc(1, sizeof(Apply));

	// set our Op operations
	OpBase_Init((OpBase *)op, OPType_APPLY, "Apply", ApplyInit, ApplyConsume,
			ApplyReset, NULL, ApplyClone, ApplyFree, false, plan);

	return (OpBase *)op;
}

static OpResult ApplyInit
(
	OpBase *opBase
) {
	ASSERT(opBase->childCount == 2);

	Apply *op = (Apply *)opBase;
	op->rhs_branch   = opBase->children[1];
	op->bound_branch = opBase->children[0];

	return OP_OK;
}

static Record ApplyConsume
(
	OpBase *opBase
) {
	Apply *op = (Apply *)opBase;

	while(true) {
		if(op->r == NULL) {
			// retrieve a record from the bound branch
			op->r = OpBase_Consume(op->bound_branch);
			if(op->r == NULL) {
				return NULL; // bound branch and this op are depleted
			}
		}

		// pull a Record from the RHS branch
		Record rhs_record = OpBase_Consume(op->rhs_branch);

		if(rhs_record == NULL) {
			// RHS branch depleted for the current bound record
			// free it and loop back to retrieve a new one
			OpBase_DeleteRecord(op->r);
			op->r = NULL;

			// reset the RHS branch
			OpBase_PropagateReset(op->rhs_branch);
			continue;
		}

		// merge bound branch record into retrieved RHS branch record
		// and return it
		Record_Merge(rhs_record, op->r);
		return rhs_record;
	}

	// not suppose to reach this point
	return NULL;
}

static OpResult ApplyReset
(
	OpBase *opBase
) {
	Apply *op = (Apply *)opBase;

	if(op->r != NULL) OpBase_DeleteRecord(op->r);
	op->r = NULL;

	return OP_OK;
}

static OpBase *ApplyClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	return NewApplyOp(plan);
}

static void ApplyFree
(
	OpBase *opBase
) {
	Apply *op = (Apply *)opBase;

	if(op->r != NULL) OpBase_DeleteRecord(op->r);
	op->r = NULL;
}

