/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "utils.h"
#include "../../query_ctx.h"
#include "../algebraic_expression.h"

Delta_Matrix _Eval_Add
(
	const AlgebraicExpression *exp,
	Delta_Matrix res
) {
	ASSERT(exp);
	ASSERT(AlgebraicExpression_ChildCount(exp) > 1);

	GrB_Info info;
	UNUSED(info);
	GrB_Index nrows;                   // number of rows of operand
	GrB_Index ncols;                   // number of columns of operand

	bool           res_in_use  =  false;  //  can we use `res` for intermediate evaluation
	Delta_Matrix   A           =  NULL;   //  left operand
	Delta_Matrix   B           =  NULL;   //  right operand
	Delta_Matrix   inter       =  NULL;   //  intermediate matrix
	GrB_Type       t           =  NULL;   //  type of operand

	// get left and right operands
	AlgebraicExpression *left = CHILD_AT(exp, 0);
	AlgebraicExpression *right = CHILD_AT(exp, 1);

	// if left operand is a matrix, simply get it
	// otherwise evaluate left hand side using `res` to store LHS value
	if(left->type == AL_OPERATION) {
		A = AlgebraicExpression_Eval(left, res);
		res_in_use = true;
	} else {
		A = left->operand.matrix;
	}

	// if right operand is a matrix, simply get it
	// otherwise evaluate right hand side using `res`
	// if free or create an additional matrix to store RHS value
	if(right->type == AL_OPERATION) {
		if(res_in_use) {
			// `res` is in use, create an additional matrix
			Delta_Matrix_nrows(&nrows, res);
			Delta_Matrix_ncols(&ncols, res);
			info = Delta_Matrix_new(&inter, GrB_BOOL, nrows, ncols, false);
			ASSERT(info == GrB_SUCCESS);
			B = AlgebraicExpression_Eval(right, inter);
		} else {
			// `res` is not used just yet, use it for RHS evaluation
			B = AlgebraicExpression_Eval(right, res);
		}
	} else {
		B = right->operand.matrix;
	}

	//--------------------------------------------------------------------------
	// perform addition
	//--------------------------------------------------------------------------
	info = Delta_add(res, A, B);
	ASSERT(info == GrB_SUCCESS);

	uint child_count = AlgebraicExpression_ChildCount(exp);
	// expression has more than 2 operands, e.g. A+B+C...
	for(uint i = 2; i < child_count; i++) {
		right = CHILD_AT(exp, i);

		if(right->type == AL_OPERAND) {
			B = right->operand.matrix;
		} else {
			// 'right' represents either + or * operation
			if(inter == NULL) {
				// can't use `res`, use an intermidate matrix
				Delta_Matrix_nrows(&nrows, res);
				Delta_Matrix_ncols(&ncols, res);
				info = Delta_Matrix_new(&inter, GrB_BOOL, nrows, ncols, false);
				ASSERT(info == GrB_SUCCESS);
			}
			AlgebraicExpression_Eval(right, inter);
			B = inter;
		}

		// TODO: handle different types
		GrB_OK (Delta_Matrix_type(&t, B));
		ASSERT (t == GrB_BOOL);

		// perform addition
		info = Delta_add(res, res, B);
		ASSERT(info == GrB_SUCCESS);
	}

	if(inter != NULL) Delta_Matrix_free(&inter);
	return res;
}

