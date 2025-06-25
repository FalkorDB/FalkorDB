/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "utils.h"
#include "../../query_ctx.h"
#include "../algebraic_expression.h"

static void _entry_present (bool *z, const bool *x, const uint64_t *y)
{
	*z = *x && *y != U64_ZOMBIE;
}
#define _ENTRY_PRESENT                                                         \
"void _entry_present (bool *z, const bool *x, const uint64_t *y)\n" \
"{\n"                                                                          \
"	*z = *x && *y !=  (1UL << (sizeof(uint64_t) * 8 - 1)) ;\n"                 \
"}\n"                                                                          \

Delta_Matrix _Eval_Mul
(
	const AlgebraicExpression *exp,
	Delta_Matrix res
) {
	// GrB_set (GrB_GLOBAL, GxB_JIT_ON, GxB_JIT_C_CONTROL) ;
	GrB_BinaryOp not_zombie = NULL;
	GrB_Semiring any_alive  = NULL;
	GxB_BinaryOp_new(
		&not_zombie, (GxB_binary_function) &_entry_present, 
		GrB_BOOL, GrB_BOOL, GrB_UINT64, "_entry_present", _ENTRY_PRESENT
	);
	GrB_Semiring_new (&any_alive, GrB_LOR_MONOID_BOOL, not_zombie);
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

	GrB_Matrix    res_m        = DELTA_MATRIX_M(res);
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
			A = DELTA_MATRIX_M(M) ;
			continue ;
		}
		Delta_Matrix_type(&ty, M);
		semiring = (ty == GrB_BOOL)? GrB_LOR_LAND_SEMIRING_BOOL: any_alive;
		// 	info = Delta_mxm(res, semiring, A, M);
		info = Delta_mxm_identity(res_m, semiring, A, M);
		ASSERT(info == GrB_SUCCESS);
		
		res_modified = true ;
		// setup for next iteration
		A = res_m ;

		// exit early if 'res' is empty 0 * A = 0
		bool alive = false;
		info = GrB_Matrix_reduce_BOOL(
			&alive, NULL, GrB_LOR_MONOID_BOOL, res_m, NULL);
		ASSERT(info == GrB_SUCCESS) ;
		if(!alive) break ;
	}

	if(!res_modified) {
		GxB_Matrix_type(&ty, A);
		ASSERT(ty == GrB_BOOL);
		info = GrB_transpose(res_m, NULL, NULL, A, GrB_DESC_T0) ;
		ASSERT(info == GrB_SUCCESS) ;
	}
	if(res_modified)
	{
		GrB_Matrix res_dm = DELTA_MATRIX_DELTA_MINUS(res);
		//add any explicit zeros to the DM matrix
		info = GrB_Matrix_select_BOOL(
			res_dm, NULL, NULL, GrB_VALUEEQ_BOOL, res_m, BOOL_ZOMBIE, NULL);
		ASSERT(info == GrB_SUCCESS) ;
		Delta_Matrix_wait(res, false);
	}

	GrB_free(&not_zombie);
	GrB_free(&any_alive);
	// GrB_set (GrB_GLOBAL, GxB_JIT_LOAD, GxB_JIT_C_CONTROL) ;

	return res ;
}

