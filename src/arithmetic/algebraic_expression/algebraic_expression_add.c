/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "utils.h"
#include "../../query_ctx.h"
#include "../algebraic_expression.h"

RG_Matrix _Eval_Add
(
	const AlgebraicExpression *exp,
	RG_Matrix res
) {
	ASSERT(exp);
	ASSERT(AlgebraicExpression_ChildCount(exp) > 1);

	GrB_Info info;
	UNUSED(info);
	GrB_Index nrows;                   // number of rows of operand
	GrB_Index ncols;                   // number of columns of operand

	bool        res_in_use  =  false;  //  can we use `res` for intermediate evaluation
	RG_Matrix   A           =  NULL;   //  left operand
	RG_Matrix   B           =  NULL;   //  right operand
	RG_Matrix   inter       =  NULL;   //  intermediate matrix

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
			RG_Matrix_nrows(&nrows, res);
			RG_Matrix_ncols(&ncols, res);
			info = RG_Matrix_new(&inter, GrB_BOOL, nrows, ncols);
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

	info = RG_eWiseAdd(res, GxB_ANY_PAIR_BOOL, A, B);
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
				RG_Matrix_nrows(&nrows, res);
				RG_Matrix_ncols(&ncols, res);
				info = RG_Matrix_new(&inter, GrB_BOOL, nrows, ncols);
				ASSERT(info == GrB_SUCCESS);
			}
			AlgebraicExpression_Eval(right, inter);
			B = inter;
		}

		// perform addition
		info = RG_eWiseAdd(res, GxB_ANY_PAIR_BOOL, res, B);
		ASSERT(info == GrB_SUCCESS);
	}

	if(inter != NULL) RG_Matrix_free(&inter);
	return res;
}

