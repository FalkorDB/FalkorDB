/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_utils.h"
#include "delta_matrix.h"
#include "../../util/arr.h"

// C (i,j) = x
GrB_Info Delta_Matrix_setElement_UINT64
(
    Delta_Matrix C,  // matrix to modify
    uint64_t x,      // scalar to assign to C(i,j)
    GrB_Index i,     // row index
    GrB_Index j      // column index
) {
	ASSERT (C != NULL) ;
	Delta_Matrix_checkBounds (C, i, j) ;

	GrB_Info info ;
	bool already_allocated = false ;

	GrB_Matrix M  = DELTA_MATRIX_M (C) ;
	GrB_Matrix DP = DELTA_MATRIX_DELTA_PLUS (C) ;
	GrB_Matrix DM = DELTA_MATRIX_DELTA_MINUS (C) ;

#ifdef RG_DEBUG
	//--------------------------------------------------------------------------
	// validate type
	//--------------------------------------------------------------------------

	GrB_Type t ;
	info = GxB_Matrix_type (&t, M) ;
	ASSERT (info == GrB_SUCCESS) ;
	ASSERT (t == GrB_UINT64) ;
#endif

	if (DELTA_MATRIX_MAINTAIN_TRANSPOSE (C)) {
		GrB_Matrix TM  = DELTA_MATRIX_M (C->transposed) ;
		GrB_Matrix TDP = DELTA_MATRIX_DELTA_PLUS (C->transposed) ;
		GrB_Matrix TDM = DELTA_MATRIX_DELTA_MINUS (C->transposed) ;

		GrB_OK (info = GxB_Matrix_isStoredElement (TM, j, i)) ;
		already_allocated = (info == GrB_SUCCESS);

		if (already_allocated) {
			// unset delta-minus
			GrB_OK (GrB_Matrix_removeElement (TDM, j, i)) ;
		} else {
			// update entry to dp[i, j]
			GrB_OK (GrB_Matrix_setElement_BOOL (TDP, true, j, i)) ;
		}
	} else {
		GrB_OK (info = GxB_Matrix_isStoredElement (M, i, j)) ;
		already_allocated = (info == GrB_SUCCESS);
	}

	if (already_allocated) {
		// unset delta-minus
		GrB_OK (GrB_Matrix_removeElement (DM, i, j)) ;

		// overwrite m[i,j]
		info = GrB_Matrix_setElement (M, x, i, j) ;
		ASSERT (info == GrB_SUCCESS) ;
	} else {
		// update entry to dp[i, j]
		info = GrB_Matrix_setElement_UINT64 (DP, x, i, j) ;
		ASSERT (info == GrB_SUCCESS) ;
	}

	Delta_Matrix_setDirty (C) ;
	return info ;
}

