/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_apply_multiplexer.h"
#include "../execution_plan_build/execution_plan_util.h"

// forward declerations
static OpResult OpApplyMultiplexerInit(OpBase *opBase);
static Record OrMultiplexer_Consume(OpBase *opBase);
static Record AndMultiplexer_Consume(OpBase *opBase);
static OpResult OpApplyMultiplexerReset(OpBase *opBase);
static OpBase *OpApplyMultiplexerClone(const ExecutionPlan *plan, const OpBase *opBase);
static void OpApplyMultiplexerFree(OpBase *opBase);

static Record _pullFromBranchStream
(
	OpApplyMultiplexer *op,
	int branch_index
) {
	// propegate record to the top of the match stream
	Argument_AddRecord(op->branch_arguments[branch_index - 1],
			OpBase_CloneRecord(op->r));

	return OpBase_Consume(op->op.children[branch_index]);
}

OpBase *NewApplyMultiplexerOp
(
	const ExecutionPlan *plan,
	AST_Operator boolean_operator
) {

	OpApplyMultiplexer *op = rm_calloc(1, sizeof(OpApplyMultiplexer));
	op->boolean_operator = boolean_operator;

	// set our Op operations
	if(boolean_operator == OP_OR) {
		OpBase_Init((OpBase *)op, OPType_OR_APPLY_MULTIPLEXER,
				"OR Apply Multiplexer", OpApplyMultiplexerInit,
				OrMultiplexer_Consume, OpApplyMultiplexerReset, NULL,
				OpApplyMultiplexerClone, OpApplyMultiplexerFree, false, plan);
	} else if(boolean_operator == OP_AND) {
		OpBase_Init((OpBase *)op, OPType_AND_APPLY_MULTIPLEXER,
				"AND Apply Multiplexer", OpApplyMultiplexerInit,
				AndMultiplexer_Consume, OpApplyMultiplexerReset, NULL,
				OpApplyMultiplexerClone, OpApplyMultiplexerFree, false, plan);
	} else {
		ASSERT("apply multiplexer boolean operator should be AND or OR only" && false);
	}

	return (OpBase *) op;
}

// sorts the multiplexer children. Apply operations to the very end (rightmost)
// filter operations to the beginning (leftmost)
// the filter operations evaluted faster then the apply operations
// so we want to evaluate them first
static void _OpApplyMultiplexer_SortChildren
(
	OpBase *op
) {
	// the 0 child is the bounded branch of this operation
	// which consumes the record and is set on execution_plan_reduce_to_apply.c
	for(int i = 1; i < op->childCount; i++) {
		OpBase *child = op->children[i];
		// push apply ops to the end
		if(OP_IS_APPLY(child)) {
			// from current position to the end, search for filter op
			bool swapped = false;
			for(int j = i + 1; j < op->childCount; j++) {
				OpBase *candidate = op->children[j];
				if(candidate->type == OPType_FILTER) {
					op->children[i] = candidate;
					op->children[j] = child;
					swapped = true;
					break;
				}
			}
			// no swap occurred, everything is sorted
			if(!swapped) return;
		}
	}
}

// in this init function, the children 1..n are sorted such that branches which
// are filter ops will be placed at the begining of the children array
// and branches which are apply ops will be placed at the end of
// the children array since filters are easier to evalute
// for each branch the init function will collect it argument op
// for the injection of the bounded branch record
static OpResult OpApplyMultiplexerInit
(
	OpBase *opBase
) {
	// sort children
	_OpApplyMultiplexer_SortChildren(opBase);
	OpApplyMultiplexer *apply_multiplexer = (OpApplyMultiplexer *) opBase;

	// set up bounded branch
	// the bounded branch is set as the first child during the operation
	// building procedure at execution_plan_reduce_to_apply.c
	apply_multiplexer->bound_branch = opBase->children[0];
	ASSERT(apply_multiplexer->bound_branch);
	int childCount = opBase->childCount;
	// for every child, find its argument op for record injection
	apply_multiplexer->branch_arguments = array_new(OpArgument *, childCount - 1);
	for(int i = 1; i < childCount; i++) {
		OpBase *child = opBase->children[i];
		OpArgument *arg = (OpArgument *)ExecutionPlan_LocateOp(child, OPType_ARGUMENT);
		ASSERT(arg);
		array_append(apply_multiplexer->branch_arguments, arg);
	}
	return OP_OK;
}

static Record OrMultiplexer_Consume
(
	OpBase *opBase
) {
	OpApplyMultiplexer *op = (OpApplyMultiplexer *)opBase;
	while(true) {
		// try to get a record from bound stream
		op->r = OpBase_Consume(op->bound_branch);
		if(!op->r) return NULL; // depleted

		// try to get a record from some stream
		for(int i = 1; i < op->op.childCount; i++) {
			Record branch_record = _pullFromBranchStream(op, i);
			if(branch_record) {
				// don't care about the branch record
				OpBase_DeleteRecord(&branch_record);
				Record r = op->r;
				op->r = NULL;   // null to avoid double free
				return r;
			}
		}
		// did not managed to get a record from any branch
		// loop back and restart
		OpBase_DeleteRecord(&op->r);
	}
}

static Record AndMultiplexer_Consume
(
	OpBase *opBase
) {
	OpApplyMultiplexer *op = (OpApplyMultiplexer *)opBase;
	while(true) {
		// try to get a record from bound stream
		op->r = OpBase_Consume(op->bound_branch);
		if(!op->r) return NULL; // depleted

		// try to get a record from some stream
		for(int i = 1; i < op->op.childCount; i++) {
			Record branch_record = _pullFromBranchStream(op, i);
			// don't care about the branch record
			if(branch_record) OpBase_DeleteRecord(&branch_record);
			else {
				// did not managed to get a record from some branch
				// loop back and restart
				OpBase_DeleteRecord(&op->r);
				break;
			}
		}
		// all branches returned record =>
		// all filters are satisfied by the bounded record
		Record r = op->r;
		op->r = NULL;  // Null to avoid double free
		return r;
	}
}

static OpResult OpApplyMultiplexerReset
(
	OpBase *opBase
) {
	OpApplyMultiplexer *op = (OpApplyMultiplexer *)opBase;

	if(op->r) {
		OpBase_DeleteRecord(&op->r);
	}

	return OP_OK;
}

static inline OpBase *OpApplyMultiplexerClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_OR_APPLY_MULTIPLEXER ||
		   opBase->type == OPType_AND_APPLY_MULTIPLEXER);

	OpApplyMultiplexer *op = (OpApplyMultiplexer *)opBase;

	return NewApplyMultiplexerOp(plan, op->boolean_operator);
}

static void OpApplyMultiplexerFree
(
	OpBase *opBase
) {
	OpApplyMultiplexer *op = (OpApplyMultiplexer *)opBase;

	if(op->branch_arguments) {
		array_free(op->branch_arguments);
		op->branch_arguments = NULL;
	}

	if(op->r) {
		OpBase_DeleteRecord(&op->r);
	}
}

