/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "rg_utils.h"
#include "rg_matrix.h"

GrB_Info RG_Matrix_setElement_BOOL      // C (i,j) = x
(
    RG_Matrix C,                        // matrix to modify
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) {
	ASSERT(C != NULL);
	ASSERT(!RG_MATRIX_MULTI_EDGE(C));
	RG_Matrix_checkBounds(C, i, j);

	bool v;
	GrB_Info info;

	GrB_Matrix m  = RG_MATRIX_M(C);
	GrB_Matrix dp = RG_MATRIX_DELTA_PLUS(C);
	GrB_Matrix dm = RG_MATRIX_DELTA_MINUS(C);

	bool  already_allocated    =  false;  // M[i,j] exists
	bool  marked_for_deletion  =  false;  // dm[i,j] exists

	if(RG_MATRIX_MAINTAIN_TRANSPOSE(C)) {
		info = RG_Matrix_setElement_BOOL(C->transposed, j, i);
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

	RG_Matrix_setDirty(C);

	return info;
}

