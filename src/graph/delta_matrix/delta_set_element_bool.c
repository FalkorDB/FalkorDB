/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_utils.h"
#include "delta_matrix.h"

// C (i,j) = x
GrB_Info Delta_Matrix_setElement_BOOL   
(
    Delta_Matrix C,  // matrix to modify
    GrB_Index i,     // row index
    GrB_Index j      // column index
) {
	ASSERT(C != NULL);
	ASSERT(!DELTA_MATRIX_MULTI_EDGE(C));
	Delta_Matrix_checkBounds(C, i, j);
	GrB_Info info;
	bool v;

	GrB_Matrix m  = DELTA_MATRIX_M(C);
	GrB_Matrix dp = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix dm = DELTA_MATRIX_DELTA_MINUS(C);

	bool  already_allocated    =  false;  // M[i,j] exists
	bool  marked_for_deletion  =  false;  // dm[i,j] exists

	if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(C)) {
		GrB_OK(Delta_Matrix_setElement_BOOL(C->transposed, j, i));
	}

	GrB_OK(info = GxB_Matrix_isStoredElement(dm, i, j));

	marked_for_deletion = (info == GrB_SUCCESS);
	if(marked_for_deletion) {
		// unset delta-minus. assign m to true
		GrB_OK(GrB_Matrix_setElement(m, true, i, j));
		GrB_OK(GrB_Matrix_removeElement(dm, i, j));
	} else {
		GrB_OK(info = GxB_Matrix_isStoredElement(m, i, j));
		already_allocated = (info == GrB_SUCCESS);

		if(!already_allocated) {
			// update entry to dp[i, j]
			GrB_OK(GrB_Matrix_setElement_BOOL(dp, true, i, j));
		}
	}

	Delta_Matrix_setDirty(C);
	return GrB_SUCCESS;
}

