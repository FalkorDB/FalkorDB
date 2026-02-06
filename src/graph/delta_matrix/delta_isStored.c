/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"

// check if element A(i,j) is stored in the delta matrix
GrB_Info Delta_Matrix_isStoredElement
(
	const Delta_Matrix A,  // matrix to check
	GrB_Index i,           // row index
	GrB_Index j            // column index
) {
	ASSERT (A != NULL) ;

	GrB_Info info;
	bool in_M  = false;
	GrB_Matrix m  = DELTA_MATRIX_M(A);
	GrB_Matrix dp = DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix dm = DELTA_MATRIX_DELTA_MINUS(A);

	// see if entry exists in m
	GrB_OK (info = GxB_Matrix_isStoredElement(m, i, j));
	in_M = info == GrB_SUCCESS;

	if (in_M){
		// if dm[i,j] exists, return no value
		GrB_OK (info = GxB_Matrix_isStoredElement(dm, i, j));
		info = (info == GrB_NO_VALUE) ? GrB_SUCCESS : GrB_NO_VALUE ;
	} else {
		// if dp[i,j] exists return it
		GrB_OK (info = GxB_Matrix_isStoredElement(dp, i, j));
	}

	return info ;
}

