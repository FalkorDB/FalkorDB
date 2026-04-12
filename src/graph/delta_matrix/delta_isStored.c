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

	GrB_Info info ;
	GrB_Matrix M  = DELTA_MATRIX_M (A) ;
	GrB_Matrix DP = DELTA_MATRIX_DELTA_PLUS (A) ;
	GrB_Matrix DM = DELTA_MATRIX_DELTA_MINUS (A) ;
	bool in_M  = false ;
	bool in_DM = false ;

	// if dp[i,j] exists return it
	info = GxB_Matrix_isStoredElement (DP, i, j) ;
	if(info == GrB_SUCCESS) {
		return info ;
	}

	// if dm[i,j] exists, return no value
	info = GxB_Matrix_isStoredElement (DM, i, j) ;
	if (info == GrB_SUCCESS) {
		// entry marked for deletion
		return GrB_NO_VALUE ;
	}

	// entry isn't marked for deletion, see if it exists in 'm'
	info = GxB_Matrix_isStoredElement (M, i, j) ;
	return info ;
}

