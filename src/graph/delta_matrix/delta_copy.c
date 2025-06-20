/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_utils.h"
#include "delta_matrix.h"
#include "../../util/rmalloc.h"

GrB_Info Delta_Matrix_copy
(
	Delta_Matrix C,
	const Delta_Matrix A
) {
	GrB_Info    info =  GrB_SUCCESS;
	Delta_Matrix_checkCompatible(C, A);
	
	GrB_Matrix  in_m             =  DELTA_MATRIX_M(A);
	GrB_Matrix  out_m            =  DELTA_MATRIX_M(C);
	GrB_Matrix  in_delta_plus    =  DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix  in_delta_minus   =  DELTA_MATRIX_DELTA_MINUS(A);
	GrB_Matrix  out_delta_plus   =  DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix  out_delta_minus  =  DELTA_MATRIX_DELTA_MINUS(C);

	info = GrB_transpose(
		out_m, NULL, NULL, in_m, GrB_DESC_T0);
	ASSERT(info == GrB_SUCCESS);
	info = GrB_transpose(
		out_delta_plus, NULL, NULL, in_delta_plus, GrB_DESC_T0);
	ASSERT(info == GrB_SUCCESS);
	info = GrB_transpose(
		out_delta_minus, NULL, NULL, in_delta_minus, GrB_DESC_T0);
	ASSERT(info == GrB_SUCCESS);
	return info;
}

