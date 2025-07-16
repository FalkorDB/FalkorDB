/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "op_apply.h"
#include "../execution_plan_build/execution_plan_util.h"

/* Forward declarations. */
static OpResult ApplyInit(OpBase *opBase);
static Record ApplyConsume(OpBase *opBase);
static OpResult ApplyReset(OpBase *opBase);
static OpBase *ApplyClone(const ExecutionPlan *plan, const OpBase *opBase);
static void ApplyFree(OpBase *opBase);

OpBase *NewApplyOp(const ExecutionPlan *plan) {
	Apply *op = rm_malloc(sizeof(Apply));

	op->r            = NULL;
	op->op_arg       = NULL;
	op->records      = NULL;
	op->rhs_branch   = NULL;
	op->bound_branch = NULL;

	// Set our Op operations
	OpBase_Init((OpBase *)op, OPType_APPLY, "Apply", ApplyInit, ApplyConsume,
			ApplyReset, NULL, ApplyClone, ApplyFree, false, plan);

	return (OpBase *)op;
}

static OpResult ApplyInit(OpBase *opBase) {
	ASSERT(opBase->childCount == 2);

	Apply *op = (Apply *)opBase;
	// the op's bound branch and optional match branch have already been
	// built as the Apply op's first and second child ops, respectively
	op->records      = array_new(Record, 1);
	op->rhs_branch   = opBase->children[1];
	op->bound_branch = opBase->children[0];

	// locate branch's Argument op tap
	op->op_arg = (Argument *)ExecutionPlan_LocateOp(op->rhs_branch,
			OPType_ARGUMENT);

	return OP_OK;
}

static Record ApplyConsume(OpBase *opBase) {
	Apply *op = (Apply *)opBase;

	while(true) {
		if(op->r == NULL) {
			// retrieve a Record from the bound branch
			op->r = OpBase_Consume(op->bound_branch);
			if(op->r == NULL) {
				return NULL; // Bound branch and this op are depleted.
			}

			// collect record for future freeing
			array_append(op->records, op->r);

			// Successfully pulled a new Record, propagate to the top of the RHS branch.
			if(op->op_arg) {
				Argument_AddRecord(op->op_arg, OpBase_CloneRecord(op->r));
			}
		}

		// pull a Record from the RHS branch
		Record rhs_record = OpBase_Consume(op->rhs_branch);

		if(rhs_record == NULL) {
			// RHS branch depleted for the current bound Record
			// free it and loop back to retrieve a new one
			op->r = NULL;
			// reset the RHS branch
			OpBase_PropagateReset(op->rhs_branch);
			continue;
		}

		// clone the bound Record and merge the RHS Record into it
		Record r = OpBase_CloneRecord(op->r);
		Record_Merge(r, rhs_record);
		// delete the RHS record, as it has been merged into r
		OpBase_DeleteRecord(rhs_record);

		return r;
	}

	return NULL;
}

static OpResult ApplyReset(OpBase *opBase) {
	Apply *op = (Apply *)opBase;
	op->r = NULL;

	// free collected records
	uint32_t n = array_len(op->records);
	for(uint32_t i = 0; i < n; i++) {
		OpBase_DeleteRecord(op->records[i]);
	}
	array_clear(op->records);

	return OP_OK;
}

static OpBase *ApplyClone(const ExecutionPlan *plan, const OpBase *opBase) {
	return NewApplyOp(plan);
}

static void ApplyFree(OpBase *opBase) {
	Apply *op = (Apply *)opBase;

	// free collected records
	if(op->records != NULL) {
		uint32_t n = array_len(op->records);
		for(uint32_t i = 0; i < n; i++) {
			OpBase_DeleteRecord(op->records[i]);
		}

		array_free(op->records);
		op->records = NULL;
	}

	op->r = NULL;
}

