/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_utils.h"
#include "delta_matrix.h"
#include "../../util/rmalloc.h"

static void _copyMatrix
(
	const GrB_Matrix in,
	GrB_Matrix out
) {
	GrB_Index   nvals;
	GrB_Info    info =  GrB_SUCCESS;

	UNUSED(info);

	info = GrB_Matrix_nvals(&nvals, in);
	ASSERT(info == GrB_SUCCESS);

	if(nvals > 0) {
		info = GrB_Matrix_apply(out, NULL, NULL, GrB_IDENTITY_BOOL, in,
				GrB_DESC_R);
	}
	else {
		GrB_Matrix_clear(out);
	}

	ASSERT(info == GrB_SUCCESS);
}

GrB_Info Delta_Matrix_copy
(
	Delta_Matrix C,
	const Delta_Matrix A
) {
	Delta_Matrix_checkCompatible(C, A);
	
	GrB_Matrix  in_m             =  DELTA_MATRIX_M(A);
	GrB_Matrix  out_m            =  DELTA_MATRIX_M(C);
	GrB_Matrix  in_delta_plus    =  DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix  in_delta_minus   =  DELTA_MATRIX_DELTA_MINUS(A);
	GrB_Matrix  out_delta_plus   =  DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix  out_delta_minus  =  DELTA_MATRIX_DELTA_MINUS(C);

	_copyMatrix(in_m, out_m);
	_copyMatrix(in_delta_plus, out_delta_plus);
	_copyMatrix(in_delta_minus, out_delta_minus);

	return GrB_SUCCESS;
}

