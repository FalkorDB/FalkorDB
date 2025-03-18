/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_apply.h"
#include "../execution_plan_build/execution_plan_util.h"

// forward declarations
static OpResult ApplyInit(OpBase *opBase);
static Record ApplyConsume(OpBase *opBase);
static OpResult ApplyReset(OpBase *opBase);
static OpBase *ApplyClone(const ExecutionPlan *plan, const OpBase *opBase);
static void ApplyFree(OpBase *opBase);

OpBase *NewApplyOp
(
	const ExecutionPlan *plan
) {
	OpApply *op = rm_malloc(sizeof(OpApply));

	op->r            = NULL;
	op->op_arg       = NULL;
	op->rhs_branch   = NULL;
	op->bound_branch = NULL;
	op->rhs_args     = array_new(OpArgument*, 1);

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

	OpApply *op = (OpApply *)opBase;
	// the op's bound branch and optional match branch have already been
	// built as the Apply op's first and second child ops, respectively
	op->bound_branch = opBase->children[0];
	op->rhs_branch   = opBase->children[1];

	// locate all reachable Argument ops
	// do not recurse into other Apply ops right hand branches
	OpBase **queue = array_new(OpBase*, 1);

	// start traversal from op's right hand side
	array_append(queue, OpBase_GetChild(opBase, 1));

	while(array_len(queue) > 0) {
		OpBase *current = array_pop(queue);

		OPType t = OpBase_Type(current);

		// found an argument op, add it to our arguments array
		if(t == OPType_ARGUMENT) {
			array_append(op->rhs_args, (OpArgument*)current);
			continue;
		}

		// only consider Apply's left hand side
		uint n = (t == OPType_APPLY) ? 1 : OpBase_ChildCount(current);

		// add child op's to queue
		for(uint i = 0; i < n; i++) {
			array_append(queue, OpBase_GetChild(current, i));
		}
	}

	op->nargs = array_len(op->rhs_args);

	array_free(queue);

	return OP_OK;
}

static Record ApplyConsume
(
	OpBase *opBase
) {
	OpApply *op = (OpApply *)opBase;

pull_lhs:
	// get a record from the left hand side branch
	if(op->r == NULL) {
		// retrieve a Record from the bound branch
		op->r = OpBase_Consume(op->bound_branch);
		if(op->r == NULL) {
			return NULL; // bound branch and this op are depleted
		}

		// successfully pulled a new record
		// propagate to the top of the RHS branch
		for(uint i = 0; i < op->nargs; i++) {
			OpArgument *arg = op->rhs_args[i];
			Argument_AddRecord(arg, OpBase_CloneRecord(op->r));
		}
	}

	// pull a Record from the RHS branch
	Record rhs_record = NULL;
	while((rhs_record = OpBase_Consume(op->rhs_branch)) != NULL) {
		// clone the bound Record and merge the RHS Record into it
		Record r = OpBase_CloneRecord(op->r);
		OpBase_MergeRecords(r, &rhs_record);
		return r;
	}

	// RHS branch depleted for the current bound Record
	// free it and loop back to retrieve a new one
	OpBase_DeleteRecord(&op->r);

	// reset the RHS branch
	OpBase_PropagateReset(op->rhs_branch);

	// try getting a new left hand side record
	goto pull_lhs;
}

static OpResult ApplyReset
(
	OpBase *opBase
) {
	OpApply *op = (OpApply *)opBase;
	if(op->r != NULL) {
		OpBase_DeleteRecord(&op->r);
	}

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
	OpApply *op = (OpApply *)opBase;

	if(op->r != NULL) {
		OpBase_DeleteRecord(&op->r);
	}

	if(op->rhs_args != NULL) {
		array_free(op->rhs_args);
		op->rhs_args = NULL;
	}
}

