/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "range_iter.h"
#include "../RG.h"

#include <string.h>

void RangeIter_free
(
	RangeIter *iter
) {
	ASSERT(iter != NULL);
	iter->active = false;
	iter->start = 0;
	iter->end = 0;
	iter->current = 0;
	iter->step = 0;
	iter->depleted = false;
}

void RangeIter_reset
(
	RangeIter *iter
) {
	ASSERT(iter != NULL);
	ASSERT(iter->active);
	iter->current = iter->start;
	iter->depleted =  (iter->end > iter->start && iter->step < 0)
	               || (iter->end < iter->start && iter->step > 0);
}

bool RangeIter_fromRangeExp
(
	RangeIter *iter,
	const AR_ExpNode *exp
) {
	ASSERT(iter != NULL);

	RangeIter_free(iter);

	if(!AR_EXP_IsOperation(exp)) {
		return false;
	}

	if(strcmp(AR_EXP_GetFuncName(exp), "range") != 0) {
		return false;
	}

	int arg_count = exp->op.child_count;
	if(arg_count != 2 && arg_count != 3) {
		return false;
	}

	AR_ExpNode *start_exp = exp->op.children[0];
	AR_ExpNode *end_exp = exp->op.children[1];
	if(!AR_EXP_IsConstant(start_exp) || !AR_EXP_IsConstant(end_exp)) {
		return false;
	}

	SIValue start_val = start_exp->operand.constant;
	SIValue end_val = end_exp->operand.constant;
	if(SI_TYPE(start_val) != T_INT64 || SI_TYPE(end_val) != T_INT64) {
		return false;
	}

	int64_t step = 1;
	if(arg_count == 3) {
		AR_ExpNode *step_exp = exp->op.children[2];
		if(!AR_EXP_IsConstant(step_exp)) {
			return false;
		}

		SIValue step_val = step_exp->operand.constant;
		if(SI_TYPE(step_val) != T_INT64) {
			return false;
		}

		step = step_val.longval;
		if(step == 0) {
			return false;
		}
	}

	iter->start   = start_val.longval;
	iter->end     = end_val.longval;
	iter->current = start_val.longval;
	iter->step    = step;
	iter->active  = true;
	RangeIter_reset(iter);
	return true;
}

bool RangeIter_next
(
	RangeIter *iter,
	int64_t *value
) {
	ASSERT(iter != NULL);
	ASSERT(iter->active);

	if(iter->depleted) {
		return false;
	}

	if(value != NULL) {
		*value = iter->current;
	}

	if(iter->current == iter->end) {
		iter->depleted = true;
	} else {
		iter->current += iter->step;
		iter->depleted =  (iter->step > 0 && iter->current > iter->end)
		               || (iter->step < 0 && iter->current < iter->end);
	}

	return true;
}
