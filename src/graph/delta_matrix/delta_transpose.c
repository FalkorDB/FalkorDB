/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"
#include "delta_utils.h"

// allocate and update the transpose of a matrix whose transpose is NULL
GrB_Info Delta_Matrix_cacheTranspose
(
	Delta_Matrix A // matrix to give transpose
) {
	ASSERT(A != NULL);
	ASSERT(A->transposed == NULL);

	GrB_Index nrows = 0;
	GrB_Index ncols = 0;
	GrB_Type  type  = NULL;

	GrB_OK(Delta_Matrix_nrows(&nrows, A));
	GrB_OK(Delta_Matrix_ncols(&ncols, A));

	GrB_OK(Delta_Matrix_type(&type, A));

	GrB_OK(Delta_Matrix_new(&A->transposed, GrB_BOOL, ncols, nrows, false));

	Delta_Matrix T   = A->transposed;
	GrB_Matrix   M   = DELTA_MATRIX_M(A);
	GrB_Matrix   DP  = DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix   DM  = DELTA_MATRIX_DELTA_MINUS(A);
	GrB_Matrix   Mt  = DELTA_MATRIX_M(T);
	GrB_Matrix   DPt = DELTA_MATRIX_DELTA_PLUS(T);
	GrB_Matrix   DMt = DELTA_MATRIX_DELTA_MINUS(T);

	GrB_OK(GrB_transpose(Mt, NULL, NULL, M, NULL));
	GrB_OK(GrB_transpose(DPt, NULL, NULL, DP, NULL));
	GrB_OK(GrB_transpose(DMt, NULL, NULL, DM, NULL));

	// ensure all matricies are iso
	GrB_OK(GrB_Matrix_apply(Mt, NULL, NULL, GxB_ONE_BOOL, Mt, NULL));
	GrB_OK(GrB_Matrix_apply(DPt, NULL, NULL, GxB_ONE_BOOL, DPt, NULL));
	GrB_OK(GrB_Matrix_apply(DMt, NULL, NULL, GxB_ONE_BOOL, DMt, NULL));

	Delta_Matrix_wait(A, false);
	return GrB_SUCCESS;
}
