/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_skip.h"
#include "../../RG.h"
#include "../../errors/errors.h"
#include "../../arithmetic/arithmetic_expression.h"

// forward declarations
static RecordBatch SkipConsume(OpBase *opBase);
static OpResult SkipReset(OpBase *opBase);
static void SkipFree(OpBase *opBase);
static OpBase *SkipClone(const ExecutionPlan *plan, const OpBase *opBase);

static void _eval_skip
(
	OpSkip *op, AR_ExpNode *skip_exp
) {
	// store a copy of the original expression
	// this is required in the case of a parameterized skip: "SKIP $L"
	// evaluating the expression will modify it replacing the parameter
	// with a constant as a result clones of this operation would invalidly
	// resolve to an outdated constant
	op->skip_exp = AR_EXP_Clone(skip_exp);

	// evaluate using the input expression
	// leaving the stored expression untouched
	SIValue s = AR_EXP_Evaluate(skip_exp, NULL);

	// validate that the skip value is numeric and non-negative.
	if(SI_TYPE(s) != T_INT64 || SI_GET_NUMERIC(s) < 0) {
		ErrorCtx_SetError(EMSG_OPERATE_ON_NON_NEGATIVE_INT, "Skip");
	}

	op->skip = SI_GET_NUMERIC(s);

	// free the expression we've evaluated
	AR_EXP_Free(skip_exp);
}

OpBase *NewSkipOp
(
	const ExecutionPlan *plan,
	AR_ExpNode *skip_exp
) {
	OpSkip *op = rm_calloc (1, sizeof(OpSkip)) ;

	_eval_skip(op, skip_exp);

	// set operations
	OpBase_Init((OpBase *)op, OPType_SKIP, "Skip", NULL, SkipConsume, SkipReset,
			NULL, SkipClone, SkipFree, false, plan);

	return (OpBase *)op;
}

static RecordBatch SkipConsume
(
	OpBase *opBase
) {
	OpSkip *op = (OpSkip *)opBase;
	OpBase *child = op->op.children[0] ;

	// as long as we're required to skip
	while (op->skipped < op->skip) {
		RecordBatch batch = OpBase_Consume (child) ;

		// depleted
		if (batch == NULL) {
			return NULL ;
		}

		size_t batch_size = RecordBatch_Size (batch) ;
		size_t remaining = op->skip - op->skipped ;

		// full batch skip
		if (batch_size <= remaining) {
			RecordBatch_Free (&batch) ;
			op->skipped += batch_size ;
		}

		else {
			// partial batch skip
			RecordBatch_DeleteFirstN (batch, remaining) ;
			op->skipped += remaining ;
			return batch ;
		}
	}

	// pass-through
	return OpBase_Consume (child) ;
}

static OpResult SkipReset
(
	OpBase *ctx
) {
	OpSkip *skip = (OpSkip *)ctx ;
	skip->skipped = 0 ;

	return OP_OK ;
}

static inline OpBase *SkipClone(const ExecutionPlan *plan, const OpBase *opBase) {
	ASSERT(opBase->type == OPType_SKIP);

	OpSkip *op = (OpSkip *)opBase;
	/* Clone the skip expression stored on the ExecutionPlan,
	 * as we don't want to modify the templated ExecutionPlan
	 * (which may occur if this expression is a parameter). */
	AR_ExpNode *skip_exp = AR_EXP_Clone(op->skip_exp);
	return NewSkipOp(plan, skip_exp);
}

static void SkipFree(OpBase *opBase) {
	OpSkip *op = (OpSkip *)opBase;

	if(op->skip_exp != NULL) {
		AR_EXP_Free(op->skip_exp);
		op->skip_exp = NULL;
	}
}

