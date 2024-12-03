/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_semi_apply.h"
#include "../execution_plan.h"
#include "../execution_plan_build/execution_plan_util.h"

// Forward declarations.
static OpResult SemiApplyInit(OpBase *opBase);
static Record SemiApplyConsume(OpBase *opBase);
static Record AntiSemiApplyConsume(OpBase *opBase);
static OpResult SemiApplyReset(OpBase *opBase);
static OpBase *SemiApplyClone(const ExecutionPlan *plan, const OpBase *opBase);
static void SemiApplyFree(OpBase *opBase);

static inline Record _pullFromMatchStream(OpSemiApply *op) {
	return OpBase_Consume(op->match_branch);
}

OpBase *NewSemiApplyOp
(
	const ExecutionPlan *plan,
	bool anti
) {
	OpSemiApply *op = rm_malloc(sizeof(OpSemiApply));

	op->r            = NULL;
	op->op_arg       = NULL;
	op->bound_branch = NULL;
	op->match_branch = NULL;

	// set our Op operations
	if(anti) {
		OpBase_Init((OpBase *)op, OPType_ANTI_SEMI_APPLY, "Anti Semi Apply", SemiApplyInit,
					AntiSemiApplyConsume, SemiApplyReset, NULL, SemiApplyClone, SemiApplyFree, false, plan);
	} else {
		OpBase_Init((OpBase *)op, OPType_SEMI_APPLY, "Semi Apply", SemiApplyInit, SemiApplyConsume,
					SemiApplyReset, NULL, SemiApplyClone, SemiApplyFree, false, plan);
	}
	return (OpBase *) op;
}

static OpResult SemiApplyInit
(
	OpBase *opBase
) {
	ASSERT(opBase->childCount == 2);

	OpSemiApply *op = (OpSemiApply *)opBase;
	/* The op bounded branch and match branch are set to be the first and second child, respectively,
	 * during the operation building procedure at execution_plan_reduce_to_apply.c */
	op->bound_branch = opBase->children[0];
	op->match_branch = opBase->children[1];
	ASSERT(op->bound_branch && op->match_branch);

	// Locate branch's Argument op tap.
	op->op_arg = (Argument *)ExecutionPlan_LocateOp(op->match_branch, OPType_ARGUMENT);
	ASSERT(op->op_arg && op->op_arg->op.childCount == 0);
	return OP_OK;
}

// this function pulls a record from the op's bounded branch, set it as an argument for the op match branch
// and consumes a record from the match branch. If there is a record from the match branch,
// the bounded branch record is returned
static Record SemiApplyConsume
(
	OpBase *opBase
) {
	OpSemiApply *op = (OpSemiApply *)opBase;
	while(true) {
		// Try to get a record from bound stream.
		op->r = OpBase_Consume(op->bound_branch);
		if(!op->r) return NULL; // Depleted.
		// Propagate Record to the top of the Match stream.
		if(op->op_arg) Argument_AddRecord(op->op_arg, OpBase_CloneRecord(op->r));

		Record rhs_record = _pullFromMatchStream(op);
		// Reset the match branch to maintain parity with the bound branch.
		OpBase_PropagateReset(op->match_branch);
		if(rhs_record) {
			/* Successfully retrieved a Record from the match stream,
			 * free it and return the bound Record. */
			OpBase_DeleteRecord(&rhs_record);
			Record r = op->r;
			op->r = NULL;   // Null to avoid double free.
			return r;
		}
		// Did not manage to get a record from right-hand side, loop back and restart.
		OpBase_DeleteRecord(&op->r);
	}
}

// this function pulls a record from the op's bounded branch, set it as an argument for the op match branch
// and consumes a record from the match branch. If there is no record from the match branch,
// the bounded branch record is returned
static Record AntiSemiApplyConsume
(
	OpBase *opBase
) {
	OpSemiApply *op = (OpSemiApply *)opBase;
	while(true) {
		// Try to get a record from bound stream.
		op->r = OpBase_Consume(op->bound_branch);
		if(!op->r) return NULL; // Depleted.

		// Propagate record to the top of the Match stream.
		// (Must clone the Record, as it will be freed in the Match stream.)
		if(op->op_arg) Argument_AddRecord(op->op_arg, OpBase_CloneRecord(op->r));
		/* Try to pull data from the right stream,
		 * returning the bound stream record if unsuccessful. */
		Record rhs_record = _pullFromMatchStream(op);
		// Reset the match branch to maintain parity with the bound branch.
		OpBase_PropagateReset(op->match_branch);
		if(rhs_record) {
			/* Successfully retrieved a Record from the match stream,
			 * free it and pull again from the bound stream. */
			OpBase_DeleteRecord(&rhs_record);
			OpBase_DeleteRecord(&op->r);
		} else {
			// Right stream returned NULL, return left handside record.
			Record r = op->r;
			op->r = NULL;   // Null to avoid double free.
			return r;
		}
	}
}

static OpResult SemiApplyReset
(
	OpBase *opBase
) {
	OpSemiApply *op = (OpSemiApply *)opBase;
	if(op->r) {
		OpBase_DeleteRecord(&op->r);
	}
	return OP_OK;
}

static inline OpBase *SemiApplyClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_SEMI_APPLY || opBase->type == OPType_ANTI_SEMI_APPLY);
	bool anti = opBase->type == OPType_ANTI_SEMI_APPLY;
	return NewSemiApplyOp(plan, anti);
}

static void SemiApplyFree
(
	OpBase *opBase
) {
	OpSemiApply *op = (OpSemiApply *)opBase;

	if(op->r) {
		OpBase_DeleteRecord(&op->r);
	}
}

