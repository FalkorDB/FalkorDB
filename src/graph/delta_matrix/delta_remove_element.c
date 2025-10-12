/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"
#include "delta_utils.h"
#include "../../util/arr.h"
#include "../../util/rmalloc.h"
#include "../../globals.h"

GrB_Info Delta_Matrix_removeElement
(
	Delta_Matrix C,  // matrix to remove entry from
	GrB_Index i,     // row index
	GrB_Index j      // column index
) {
	ASSERT(C);
	Delta_Matrix_checkBounds(C, i, j);
	GrB_Info   info;
	bool       in_m  = false;
	GrB_Matrix m     = DELTA_MATRIX_M(C);
	GrB_Matrix dp    = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix dm    = DELTA_MATRIX_DELTA_MINUS(C);

	if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(C)) {
		GrB_OK (Delta_Matrix_removeElement(C->transposed, j, i));
	}

	//--------------------------------------------------------------------------
	// find where entry is stored
	//--------------------------------------------------------------------------
	info = GxB_Matrix_isStoredElement(m, i, j);
	in_m = (info == GrB_SUCCESS);

	if(in_m) {
		// mark deletion in delta minus
		GrB_OK(GrB_Matrix_setElement_BOOL(dm, (bool) true, i, j));
	} else {
		GrB_OK (GrB_Matrix_removeElement(dp, i, j));
	}
	
	Delta_Matrix_setDirty(C);
	return GrB_SUCCESS;
}

GrB_Info Delta_Matrix_removeElements
(
	Delta_Matrix C,  // matrix to remove entry from
	GrB_Matrix A     // matrix filled with elements to remove
) {
	ASSERT(C != NULL);
	ASSERT(A != NULL);
	ASSERT(!DELTA_MATRIX_MAINTAIN_TRANSPOSE(C));

	GrB_Matrix m  = DELTA_MATRIX_M(C);
	GrB_Matrix dp = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix dm = DELTA_MATRIX_DELTA_MINUS(C);

	// find the entries that are already in M
	GrB_OK (GrB_Matrix_eWiseMult_BinaryOp(
		dm, NULL, GrB_ONEB_BOOL, GrB_ONEB_BOOL, m, A, NULL)) ;

	// remove edges in dp that are also in A.
	GrB_OK (GrB_transpose(dp, A, NULL, dp, GrB_DESC_RSCT0));

	Delta_Matrix_setDirty(C);
	return GrB_SUCCESS;
}
