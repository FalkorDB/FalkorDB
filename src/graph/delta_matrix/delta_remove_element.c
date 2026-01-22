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

// remove entry at position C[i,j]
GrB_Info Delta_Matrix_removeElement
(
	Delta_Matrix C,  // matrix to remove entry from
	GrB_Index i,     // row index
	GrB_Index j      // column index
) {
	ASSERT (C) ;
	Delta_Matrix_checkBounds (C, i, j) ;
	GrB_Info   info ;
	bool       in_m = false;
	GrB_Matrix m    = DELTA_MATRIX_M (C) ;
	GrB_Matrix dp   = DELTA_MATRIX_DELTA_PLUS (C) ;
	GrB_Matrix dm   = DELTA_MATRIX_DELTA_MINUS (C) ;

	if (DELTA_MATRIX_MAINTAIN_TRANSPOSE (C)) {
		GrB_OK (Delta_Matrix_removeElement (C->transposed, j, i)) ;
	}

	//--------------------------------------------------------------------------
	// find where entry is stored
	//--------------------------------------------------------------------------

	info = GxB_Matrix_isStoredElement (m, i, j) ;
	in_m = (info == GrB_SUCCESS) ;

	if (in_m) {
		// mark deletion in delta minus
		GrB_OK (GrB_Matrix_setElement_BOOL (dm, (bool) true, i, j)) ;
	} else {
		GrB_OK (GrB_Matrix_removeElement (dp, i, j)) ;
	}
	
	Delta_Matrix_setDirty (C) ;
	return GrB_SUCCESS ;
}

// remove all entries in matrix m from delta matrix C
GrB_Info Delta_Matrix_removeElements
(
	Delta_Matrix C,      // matrix to remove entries from
	const GrB_Matrix A,  // elements to remove
	const GrB_Matrix AT  // A's transpose
) {
	ASSERT (C != NULL) ;
	ASSERT (A != NULL) ;

	if (DELTA_MATRIX_MAINTAIN_TRANSPOSE (C)) {
		ASSERT (AT != NULL) ;
		GrB_OK (Delta_Matrix_removeElements (C->transposed, AT, NULL)) ;
	}

	GrB_Matrix m  = DELTA_MATRIX_M (C) ;
	GrB_Matrix dp = DELTA_MATRIX_DELTA_PLUS (C) ;
	GrB_Matrix dm = DELTA_MATRIX_DELTA_MINUS (C) ;

	// find the entries that are already in M and set them in DM
	GrB_OK (GrB_Matrix_eWiseMult_BinaryOp (
		dm, A, NULL, GrB_ONEB_BOOL, m, A, GrB_DESC_S)) ;

	// remove entries in DP that are also in A
	//GrB_OK (GrB_transpose (dp, A, NULL, dp, GrB_DESC_RSCT0)) ;
	GrB_Scalar s ;
	GrB_OK (GrB_Scalar_new (&s, GrB_BOOL)) ;
	GrB_OK (GrB_Matrix_assign_Scalar (dp, A, NULL, s, GrB_ALL, 0, GrB_ALL, 0,
				GrB_DESC_S)) ;
	GrB_OK (GrB_free (&s)) ;

	Delta_Matrix_setDirty (C) ;
	return GrB_SUCCESS ;
}

