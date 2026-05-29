/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "../algebraic_expression.h"

// forward declarations
Delta_Matrix _AlgebraicExpression_Eval
(
	const AlgebraicExpression *exp,
	Delta_Matrix res
);

Delta_Matrix _Eval_Mul
(
	const AlgebraicExpression *exp,
	Delta_Matrix res
);

Delta_Matrix _Eval_Add
(
	const AlgebraicExpression *exp,
	Delta_Matrix res
);

Delta_Matrix _AlgebraicExpression_Eval
(
	const AlgebraicExpression *exp,
	Delta_Matrix res
) {
	ASSERT(exp);

	// perform operation
	switch(exp->type) {
	case AL_OPERATION:
		switch(exp->operation.op) {
		case AL_EXP_MUL:
			res = _Eval_Mul(exp, res);
			break;

		case AL_EXP_ADD:
			res = _Eval_Add(exp, res);
			break;

		case AL_EXP_TRANSPOSE:
			ASSERT("transpose should have been applied prior to evaluation");
			break;

		default:
			ASSERT("Unknown algebraic expression operation" && false);
		}
		break;
	case AL_OPERAND:
		res = exp->operand.matrix;
		break;
	default:
		ASSERT("Unknown algebraic expression node type" && false);
	}

	return res;
}

Delta_Matrix AlgebraicExpression_Eval
(
	const AlgebraicExpression *exp,
	Delta_Matrix res
) {
	ASSERT(exp != NULL);
	return _AlgebraicExpression_Eval(exp, res);
}

