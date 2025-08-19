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
	const struct Global_ops *ops = Globals_GetOps();
	//--------------------------------------------------------------------------
	// validate expression
	//--------------------------------------------------------------------------

	ASSERT(exp != NULL) ;
	ASSERT(AlgebraicExpression_ChildCount(exp) > 1) ;
	ASSERT(AlgebraicExpression_OperationCount(exp, AL_EXP_MUL) == 1) ;

	GrB_Info             info;
	Delta_Matrix         M;      // current operand
	GrB_Index            nvals;  // NNZ in res
	AlgebraicExpression  *c;     // current child node
	GrB_Type             ty;
	UNUSED(info) ;

	// multiplication will work if there are deletions, but not if there are 
	// additions TODO: mxm could be made to work with additions
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
		if(!alive) break ;
	}

	if(!res_modified) {
		// copy A into res_m
		GrB_OK (GrB_transpose(res_m, NULL, NULL, A, GrB_DESC_T0)) ;
	}

	if(res_modified)
	{
		GrB_Matrix res_dm = DELTA_MATRIX_DELTA_MINUS(res);
		//add any explicit zeros to the DM matrix
		GrB_OK (GrB_Matrix_select_BOOL(
			res_dm, NULL, NULL, GrB_VALUEEQ_BOOL, res_m, BOOL_ZOMBIE, NULL));
		Delta_Matrix_wait(res, false);
	}

	return res ;
}

