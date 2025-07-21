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

RG_Matrix _Eval_Mul
(
	const AlgebraicExpression *exp,
	RG_Matrix res
) {
	//--------------------------------------------------------------------------
	// validate expression
	//--------------------------------------------------------------------------

	ASSERT(exp != NULL) ;
	ASSERT(AlgebraicExpression_ChildCount(exp) > 1) ;
	ASSERT(AlgebraicExpression_OperationCount(exp, AL_EXP_MUL) == 1) ;

	GrB_Info             info    ;
	RG_Matrix            M       ;  // current operand
	GrB_Index            nvals   ;  // NNZ in res
	AlgebraicExpression  *c      ;  // current child node

	UNUSED(info) ;

	RG_Matrix     A             =  NULL                                 ; 
	bool          res_modified  =  false                                ;
	GrB_Semiring  semiring      =  GxB_ANY_PAIR_BOOL                    ;
	uint          child_count   =  AlgebraicExpression_ChildCount(exp)  ;

	for(uint i = 0; i < child_count; i++) {
		c = CHILD_AT(exp, i) ;
		ASSERT(c->type == AL_OPERAND) ;

		M = c->operand.matrix ;

		// first time A is set
		if(A == NULL) {
			A = M ;
			continue ;
		}

		// both A and M are valid matrices, perform multiplication
		info = RG_mxm(res, semiring, A, M) ;
		res_modified = true ;
		// setup for next iteration
		A = res ;

		// exit early if 'res' is empty 0 * A = 0
		info = RG_Matrix_nvals(&nvals, res);
		ASSERT(info == GrB_SUCCESS) ;
		if(nvals == 0) break ;
	}

	if(!res_modified) {
		info = RG_Matrix_copy(res, A) ;
		ASSERT(info == GrB_SUCCESS) ;
	}

	return res ;
}

