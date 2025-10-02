/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
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
	GrB_OK (GrB_Matrix_nvals(&nvals, in));

	if(nvals > 0) {
		GrB_OK (GrB_transpose(out, NULL, NULL, in, GrB_DESC_T0));
	} else {
		GrB_OK (GrB_Matrix_clear(out));
	}
}

GrB_Info Delta_Matrix_copy
(
	Delta_Matrix C,
	const Delta_Matrix A
) {
	Delta_Matrix_checkCompatible(C, A);
	
	GrB_Matrix in_m            = DELTA_MATRIX_M(A);
	GrB_Matrix out_m           = DELTA_MATRIX_M(C);
	GrB_Matrix in_delta_plus   = DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix in_delta_minus  = DELTA_MATRIX_DELTA_MINUS(A);
	GrB_Matrix out_delta_plus  = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix out_delta_minus = DELTA_MATRIX_DELTA_MINUS(C);

	_copyMatrix(in_m, out_m);
	_copyMatrix(in_delta_plus, out_delta_plus);
	_copyMatrix(in_delta_minus, out_delta_minus);

	return GrB_SUCCESS;
}

