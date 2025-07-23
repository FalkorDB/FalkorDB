/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"
#include "delta_utils.h"
#include "../../util/arr.h"
#include "../../util/rmalloc.h"

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
		info = Delta_Matrix_removeElement_BOOL(C->transposed, j, i);
		if(info != GrB_SUCCESS) {
			return info;
		}
	}

	//--------------------------------------------------------------------------
	// find where entry is stored
	//--------------------------------------------------------------------------
	info = GxB_Matrix_isStoredElement(m, i, j);
	in_m = (info == GrB_SUCCESS);
	info = GxB_Matrix_isStoredElement(dp, i, j);
	in_dp = (info == GrB_SUCCESS);
	info = GxB_Matrix_isStoredElement(dm, i, j);
	in_dm = (info == GrB_SUCCESS);

	if(in_dm || !(in_m || in_dp)) {
		// entry already marked for deletion
		return GrB_NO_VALUE;
	}

	if(in_m) {
		// mark deletion in delta minus
		GrB_OK(GrB_Matrix_setElement(m, BOOL_ZOMBIE, i, j));
		GrB_OK(GrB_Matrix_setElement(dm, true, i, j));
		Delta_Matrix_setDirty(C);
		return GrB_SUCCESS;
	}

	//--------------------------------------------------------------------------
	// entry exists in 'delta-plus'
	//--------------------------------------------------------------------------
	
	GrB_OK (GrB_Matrix_removeElement(dp, i, j));
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
		info = Delta_Matrix_removeElement_BOOL(C->transposed, j, i);
		if(info != GrB_SUCCESS) {
			return info;
		}
	}

	//--------------------------------------------------------------------------
	// find where entry is stored
	//--------------------------------------------------------------------------
	info = GxB_Matrix_isStoredElement(m, i, j);
	in_m = (info == GrB_SUCCESS);
	info = GxB_Matrix_isStoredElement(dp, i, j);
	in_dp = (info == GrB_SUCCESS);
	info = GxB_Matrix_isStoredElement(dm, i, j);
	in_dm = (info == GrB_SUCCESS);

	if(in_dm || !(in_m || in_dp)) {
		// entry already marked for deletion
		return GrB_NO_VALUE;
	}

	if(in_m) {
		// mark deletion in delta minus
		GrB_OK(GrB_Matrix_setElement(m, U64_ZOMBIE, i, j));
		GrB_OK(GrB_Matrix_setElement(dm, true, i, j));
		Delta_Matrix_setDirty(C);
		return GrB_SUCCESS;
	}

	//--------------------------------------------------------------------------
	// entry exists in 'delta-plus'
	//--------------------------------------------------------------------------
	
	GrB_OK (GrB_Matrix_removeElement(dp, i, j));
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
	GrB_Matrix  m  =  DELTA_MATRIX_M(C);
	GrB_Matrix  dp =  DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix  dm =  DELTA_MATRIX_DELTA_MINUS(C);

	// add edges in m and A to dm
	GrB_OK(GrB_Matrix_eWiseMult_BinaryOp(
		dm, NULL, GrB_ONEB_BOOL, GrB_ONEB_BOOL, m, A, NULL)) ;	

	// remove edges in dp that are also in A.
	GrB_OK(GrB_transpose(dp, A, NULL, dp, GrB_DESC_RSCT0));

	// x XOR (x AND TRUE) - Always outputs false.
	// Done like this in the hope that GBLAS will recognize this can be done inplace.
	GrB_OK(GrB_eWiseMult(m, NULL, GrB_LXOR, GrB_LAND, m, A, NULL));

	Delta_Matrix_validate(C);
	Delta_Matrix_setDirty(C);
	return GrB_SUCCESS;
}


