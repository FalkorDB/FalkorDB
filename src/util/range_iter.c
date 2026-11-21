/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "range_iter.h"

void RangeIter_reset
(
	RangeIter *iter
) {
	ASSERT(iter != NULL);
	iter->current = iter->start;
}

bool RangeIter_fromRangeExp
(
	RangeIter *iter,
	const AR_ExpNode *exp
) {
	ASSERT(iter != NULL);
	ASSERT(exp  != NULL);

	if(!AR_EXP_IsOperation(exp)) {
		return false;
	}

	if(strcmp(AR_EXP_GetFuncName(exp), "range") != 0) {
		return false;
	}

	int arg_count = AR_EXP_getChildCount (exp);
	if(arg_count < 2 || arg_count > 3) {
		return false;
	}

	AR_ExpNode *start_exp = AR_EXP_getChild(exp, 0) ;
	AR_ExpNode *end_exp   = AR_EXP_getChild(exp, 1) ;

	if(!AR_EXP_IsConstant(start_exp) || !AR_EXP_IsConstant(end_exp)) {
		return false;
	}

	SIValue start_val = AR_EXP_Evaluate(start_exp, NULL);
	SIValue end_val   = AR_EXP_Evaluate(end_exp, NULL);

	if(SI_TYPE(start_val) != T_INT64 || SI_TYPE(end_val) != T_INT64) {
		return false;
	}

	int64_t step = 1;
	if(arg_count == 3) {
		AR_ExpNode *step_exp = AR_EXP_getChild(exp, 2) ;
		if(!AR_EXP_IsConstant(step_exp)) {
			return false;
		}

		SIValue step_val   = AR_EXP_Evaluate(step_exp, NULL);
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
	return true;
}

bool RangeIter_next
(
	RangeIter *iter,
	int64_t *value
) {
	ASSERT(iter != NULL);

	bool depleted = (iter->step > 0 && iter->current > iter->end)
	             || (iter->step < 0 && iter->current < iter->end);
	
	if(depleted) {
		return false;
	}

	if(value != NULL) {
		*value = iter->current;
	}

	iter->current += iter->step;
	return true;
}

// the number of next calls until the iterator is exhausted
int64_t RangeIter_len
(
	const RangeIter iter  // iterator to query
) {
	bool depleted = (iter.step > 0 && iter.current > iter.end)
	             || (iter.step < 0 && iter.current < iter.end)
	             || (iter.step == 0) ;
	return depleted ? 0 : (iter.end - iter.current) / iter.step + 1 ;
}

