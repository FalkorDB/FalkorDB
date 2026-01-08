/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_limit.h"
#include "../../RG.h"
#include "../../errors/errors.h"
#include "../../arithmetic/arithmetic_expression.h"

// forward declarations
static RecordBatch LimitConsume(OpBase *opBase);
static OpResult LimitReset(OpBase *opBase);
static void LimitFree(OpBase *opBase);
static OpBase *LimitClone(const ExecutionPlan *plan, const OpBase *opBase);

static void _eval_limit
(
	OpLimit *op,
	AR_ExpNode *limit_exp
) {
	// store a copy of the original expression
	// this is required in the case of a parameterized limit: "LIMIT $L"
	// evaluating the expression will modify it
	// replacing the parameter with a constant as a result
	// clones of this operation would invalidly resolve to an outdated constant
	op->limit_exp = AR_EXP_Clone(limit_exp);

	// evaluate using the input expression
	// leaving the stored expression untouched
	SIValue l = AR_EXP_Evaluate(limit_exp, NULL);

	// validate that the limit value is numeric and non-negative
	if(SI_TYPE(l) != T_INT64 || SI_GET_NUMERIC(l) < 0) {
		ErrorCtx_SetError(EMSG_OPERATE_ON_NON_NEGATIVE_INT, "Limit");
	}

	op->limit = SI_GET_NUMERIC(l);

	// free the expression we've evaluated
	AR_EXP_Free(limit_exp);
}

OpBase *NewLimitOp
(
	const ExecutionPlan *plan,
	AR_ExpNode *limit_exp
) {
	// validate inputs
	ASSERT(plan      != NULL);
	ASSERT(limit_exp != NULL);

	OpLimit *op = rm_calloc (1, sizeof(OpLimit)) ;

	_eval_limit(op, limit_exp);

	// set operations
	OpBase_Init((OpBase *)op, OPType_LIMIT, "Limit", NULL, LimitConsume,
			LimitReset, NULL, LimitClone, LimitFree, false, plan);

	return (OpBase *)op;
}

static RecordBatch LimitConsume
(
	OpBase *opBase
) {
	OpLimit *op = (OpLimit *)opBase ;

	// have we reached our limit?
	if (op->consumed >= op->limit) {
		return NULL ;
	}

	OpBase *child = op->op.children[0] ;
	RecordBatch batch = OpBase_Consume (child) ;

	if (unlikely (batch == NULL)) {
		return NULL ;
	}

	size_t remaining  = op->limit - op->consumed ;
	size_t batch_size = RecordBatch_Size (batch) ;

	if (unlikely (batch_size > remaining)) {
		RecordBatch_DeleteRecords (batch, batch_size - remaining) ;
		batch_size = remaining ;
	}

	// consume batch
	op->consumed += batch_size ;
	return batch ;
}

static OpResult LimitReset
(
	OpBase *ctx
) {
	OpLimit *limit = (OpLimit *)ctx;
	limit->consumed = 0;
	return OP_OK;
}

static inline OpBase *LimitClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_LIMIT);

	OpLimit *op = (OpLimit *)opBase;
	// clone the limit expression stored on the ExecutionPlan
	// as we don't want to modify the templated ExecutionPlan
	// (which may occur if this expression is a parameter)
	AR_ExpNode *limit_exp = AR_EXP_Clone(op->limit_exp);
	return NewLimitOp(plan, limit_exp);
}

static void LimitFree
(
	OpBase *opBase
) {
	OpLimit *op = (OpLimit *)opBase;

	if(op->limit_exp) {
		AR_EXP_Free(op->limit_exp);
		op->limit_exp = NULL;
	}
}

