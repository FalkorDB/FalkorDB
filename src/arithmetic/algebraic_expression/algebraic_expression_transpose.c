/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "../algebraic_expression.h"
#include "utils.h"
#include "../../util/arr.h"

/* Transpose expression
 * Wraps expression with a transpose operation
 * Transpose(exp) */
void AlgebraicExpression_Transpose
(
	AlgebraicExpression **exp    // Expression to transpose.
) {
	ASSERT(exp);
	AlgebraicExpression *root =
		AlgebraicExpression_NewOperation(AL_EXP_TRANSPOSE);
	AlgebraicExpression_AddChild(root, *exp);
	*exp = root;
}

