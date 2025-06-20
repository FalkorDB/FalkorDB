/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_utils.h"
#include "delta_matrix.h"

GrB_Info Delta_Matrix_setElement_BOOL   // C (i,j) = x
(
    Delta_Matrix C,                     // matrix to modify
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) {
	ASSERT(C != NULL);
	ASSERT(!DELTA_MATRIX_MULTI_EDGE(C));
	Delta_Matrix_checkBounds(C, i, j);

	bool v;
	GrB_Info info;

	GrB_Matrix m  = DELTA_MATRIX_M(C);
	GrB_Matrix dp = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix dm = DELTA_MATRIX_DELTA_MINUS(C);

	bool  already_allocated    =  false;  // M[i,j] exists
	bool  marked_for_deletion  =  false;  // dm[i,j] exists

	if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(C)) {
		info = Delta_Matrix_setElement_BOOL(C->transposed, j, i);
		ASSERT(info == GrB_SUCCESS);
	}

	info = GrB_Matrix_extractElement(&v, dm, i, j);
	marked_for_deletion = (info == GrB_SUCCESS);

	if(marked_for_deletion) {
		// unset delta-minus m already assign to true
		info = GrB_Matrix_removeElement(dm, i, j);
		ASSERT(info == GrB_SUCCESS);
	} else {
		info = GrB_Matrix_extractElement(&v, m, i, j);
		already_allocated = (info == GrB_SUCCESS);

		if(!already_allocated) {
			// update entry to dp[i, j]
			info = GrB_Matrix_setElement_BOOL(dp, true, i, j);
			ASSERT(info == GrB_SUCCESS);
		}
	}

	Delta_Matrix_setDirty(C);

	return info;
}

