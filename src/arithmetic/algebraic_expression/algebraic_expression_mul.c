/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "utils.h"
#include "../../query_ctx.h"
#include "../algebraic_expression.h"

static void _entry_present (bool *z, const uint64_t *x, const uint64_t *y)
{
	*z = *y != MSB_MASK;
}
#define _ENTRY_PRESENT                                                         \
"void _entry_present (bool *z, const uint64_t *x, const uint64_t *y)\n" \
"{\n"                                                                          \
"	*z = *y !=  (1UL << (sizeof(uint64_t) * 8 - 1)) ;\n"                       \
"}\n"                                                                          \

Delta_Matrix _Eval_Mul
(
	const AlgebraicExpression *exp,
	Delta_Matrix res
) {
	GrB_set (GrB_GLOBAL, GxB_JIT_ON, GxB_JIT_C_CONTROL) ;
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

	Delta_Matrix  A         = NULL;
	bool          res_modified = false;
	GrB_Semiring  semiring     = GxB_ANY_PAIR_BOOL;
	uint          child_count  = AlgebraicExpression_ChildCount(exp);

	for(uint i = 0; i < child_count; i++) {
		c = CHILD_AT(exp, i) ;
		ASSERT(c->type == AL_OPERAND) ;

		M = c->operand.matrix ;

		// first time A is set
		if(A == NULL) {
			A = M ;
			continue ;
		}
		Delta_Matrix_type(&ty, M);
		if(ty == GrB_BOOL){
			// both A and M are valid matrices, perform multiplication
			info = Delta_mxm(res, semiring, A, M);
		} else {
			// M is a tensor. Multiply excluding zombies
			info = Delta_mxm_identity(res, any_alive, A, M);
		}
		
		res_modified = true ;
		// setup for next iteration
		A = res ;

		// exit early if 'res' is empty 0 * A = 0
		info = Delta_Matrix_nvals(&nvals, res);
		ASSERT(info == GrB_SUCCESS) ;
		if(nvals == 0) break ;
	}

	if(!res_modified) {
		info = Delta_Matrix_copy(res, A) ;
		ASSERT(info == GrB_SUCCESS) ;
	}
	GrB_free(&not_zombie);
	GrB_free(&any_alive);
	GrB_set (GrB_GLOBAL, GxB_JIT_LOAD, GxB_JIT_C_CONTROL) ;

	return res ;
}

