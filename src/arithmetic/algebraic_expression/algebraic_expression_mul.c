/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "utils.h"
#include "../../query_ctx.h"
#include "../algebraic_expression.h"
#include "../../globals.h"

Delta_Matrix _Eval_Mul
(
	const AlgebraicExpression *exp,
	Delta_Matrix res
) {
	const struct GrB_ops *ops = Global_GrB_Ops_Get();
	//--------------------------------------------------------------------------
	// validate expression
	//--------------------------------------------------------------------------

	ASSERT(exp != NULL) ;
	ASSERT(AlgebraicExpression_ChildCount(exp) > 1) ;
	ASSERT(AlgebraicExpression_OperationCount(exp, AL_EXP_MUL) == 1) ;

	Delta_Matrix         M;      // current operand
	GrB_Index            nvals;  // NNZ in res
	AlgebraicExpression  *c;     // current child node
	GrB_Type             ty;

	// multiplication will work if there are deletions, but not if there are 
	// additions
	GrB_OK (GrB_Matrix_nvals(&nvals, DELTA_MATRIX_DELTA_PLUS(res)));
	ASSERT(nvals == 0);

	GrB_Matrix    res_m        = Delta_Matrix_M(res);
	GrB_Matrix    A            = NULL;
	bool          res_modified = false;
	GrB_Semiring  semiring     = NULL;
	uint          child_count  = AlgebraicExpression_ChildCount(exp);


	for(uint i = 0; i < child_count; i++) {
		c = CHILD_AT(exp, i) ;
		ASSERT(c->type == AL_OPERAND) ;

		M = c->operand.matrix ;

		// first time A is set
		if(A == NULL) {
			ASSERT(Delta_Matrix_Synced(M));
			A = Delta_Matrix_M(M) ;
			continue ;
		}

		Delta_Matrix_type(&ty, M);
		semiring = (ty == GrB_BOOL)? GrB_LOR_LAND_SEMIRING_BOOL: ops->any_alive;
		GrB_OK (Delta_mxm_identity(res_m, semiring, GxB_ANY_PAIR_BOOL, A, M));

		// info = Delta_mxm_count(res_m, GxB_PLUS_PAIR_UINT64, A, M);
		// ASSERT(info == GrB_SUCCESS);
		
		res_modified = true ;
		// setup for next iteration
		A = res_m ;

		// exit early if 'res' is empty 0 * A = 0
		bool alive = false;
		GrB_OK (GrB_Matrix_reduce_BOOL(
			&alive, NULL, GrB_LOR_MONOID_BOOL, res_m, NULL));
		if(!alive) 
		{
			GrB_OK (GrB_Matrix_clear(res_m));
			break;
		}
	}

	if(!res_modified) {
		// copy A into res_m
		GrB_OK (GrB_transpose(res_m, NULL, NULL, A, GrB_DESC_T0)) ;
	}
	
	GrB_Matrix_wait(res_m, GrB_MATERIALIZE) ;
	if(res_modified) {
		GrB_Matrix_select_BOOL(DELTA_MATRIX_DELTA_MINUS(res), NULL, NULL, 
			GrB_VALUEEQ_BOOL, res_m, BOOL_ZOMBIE, NULL);
	}

	return res ;
}

