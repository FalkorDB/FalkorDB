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

GrB_Info Delta_Matrix_removeElement_BOOL
(
	Delta_Matrix C,  // matrix to remove entry from
	GrB_Index i,     // row index
	GrB_Index j      // column index
) {
	ASSERT(C);
	Delta_Matrix_checkBounds(C, i, j);
	GrB_Info   info;
	bool       in_m  = false;
	bool       in_dp = false;
	bool       in_dm = false;
	GrB_Matrix m     = DELTA_MATRIX_M(C);
	GrB_Matrix dp    = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix dm    = DELTA_MATRIX_DELTA_MINUS(C);

#ifdef RG_DEBUG
	GrB_Type type;
	GrB_OK(GxB_Matrix_type(&type, m));
	ASSERT(type == GrB_BOOL);
#endif

	if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(C)) {
		GrB_OK (Delta_Matrix_removeElement_BOOL(C->transposed, j, i));
	}

	//--------------------------------------------------------------------------
	// find where entry is stored
	//--------------------------------------------------------------------------
	info = GxB_Matrix_isStoredElement(m, i, j);
	in_m = (info == GrB_SUCCESS);

	if(in_m) {
		// mark deletion in delta minus and M
		GrB_OK (GrB_Matrix_setElement_BOOL(m, BOOL_ZOMBIE, i, j));
		GrB_OK (GrB_Matrix_setElement_BOOL(dm, true, i, j));
	} else {
		GrB_OK (GrB_Matrix_removeElement(dp, i, j));
	}

	//--------------------------------------------------------------------------
	// entry exists in 'delta-plus'
	//--------------------------------------------------------------------------
	
	Delta_Matrix_setDirty(C);
	return GrB_SUCCESS;
}

GrB_Info Delta_Matrix_removeElement_UINT64
(
    Delta_Matrix C,  // matrix to remove entry from
    GrB_Index i,     // row index
    GrB_Index j      // column index
) {
	ASSERT(C);
	Delta_Matrix_checkBounds(C, i, j);
	GrB_Info   info;
	bool       in_m  = false;
	bool       in_dp = false;
	bool       in_dm = false;
	GrB_Matrix m     = DELTA_MATRIX_M(C);
	GrB_Matrix dp    = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix dm    = DELTA_MATRIX_DELTA_MINUS(C);

#ifdef RG_DEBUG
	GrB_Type type;
	GrB_OK(GxB_Matrix_type(&type, m));
	ASSERT(type == GrB_UINT64);
#endif
	if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(C)) {
		GrB_OK (Delta_Matrix_removeElement_BOOL(C->transposed, j, i));
	}

	//--------------------------------------------------------------------------
	// find where entry is stored
	//--------------------------------------------------------------------------
	info = GxB_Matrix_isStoredElement(m, i, j);
	in_m = (info == GrB_SUCCESS);

	if(in_m) {
		// mark deletion in delta minus and M
		GrB_OK (GrB_Matrix_setElement_UINT64(m, U64_ZOMBIE, i, j));
		GrB_OK (GrB_Matrix_setElement_BOOL(dm, true, i, j));
	} else {
		GrB_OK (GrB_Matrix_removeElement(dp, i, j));
	}
	
	//--------------------------------------------------------------------------
	// entry exists in 'delta-plus'
	//--------------------------------------------------------------------------
	
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

	GrB_Matrix  m    =  DELTA_MATRIX_M(C);
	GrB_Matrix  dp   =  DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix  dm   =  DELTA_MATRIX_DELTA_MINUS(C);
	GrB_Matrix  dels = NULL; // entries in M to mark as deleted

	GrB_Index nrows = 0;
	GrB_Index ncols = 0;

	GrB_OK (Delta_Matrix_nrows(&nrows, C));
	GrB_OK (Delta_Matrix_ncols(&ncols, C));
	GrB_OK (GrB_Matrix_new(&dels, GrB_BOOL, nrows, ncols));

	// find the entries that are already in M
	GrB_OK (GrB_Matrix_eWiseMult_BinaryOp(
		dels, NULL, NULL, GrB_ONEB_BOOL, m, A, NULL)) ;

	// add edges in m and A to dm
	GrB_OK (GrB_transpose (dm, NULL, GrB_ONEB_BOOL, dels, GrB_DESC_T0)) ;	

	// remove edges in dp that are also in A.
	GrB_OK (GrB_transpose(dp, A, NULL, dp, GrB_DESC_RSCT0));

	GrB_OK (GrB_Matrix_assign_BOOL(m, dels, NULL, BOOL_ZOMBIE, GrB_ALL, 0, 
		GrB_ALL, 0, GrB_DESC_S));
	GrB_OK (GrB_wait(m, GrB_MATERIALIZE));
	GrB_free (&dels);

	Delta_Matrix_setDirty(C);
	return GrB_SUCCESS;
}


