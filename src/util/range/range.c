/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "range.h"

RangeExpression RangeExpression_Clone
(
	RangeExpression range
) {
	return (RangeExpression){.op = range.op, .exp = AR_EXP_Clone(range.exp)};
}

void RangeExpression_Free
(
	RangeExpression *range
) {
	ASSERT(range      != NULL);
	ASSERT(range->exp != NULL);

	AR_EXP_Free(range->exp);
}

