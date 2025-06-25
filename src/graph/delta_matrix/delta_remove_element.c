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
	Delta_Matrix C,                    // matrix to remove entry from
	GrB_Index i,                    // row index
	GrB_Index j                     // column index
) {
	ASSERT(C);
	Delta_Matrix_checkBounds(C, i, j);

	bool        m_x;
	bool        dm_x;
	bool        dp_x;
	GrB_Info    info;
	GrB_Type    type;
	bool        in_m        =  false;
	bool        in_dp       =  false;
	bool        in_dm       =  false;
	GrB_Matrix  m           =  DELTA_MATRIX_M(C);
	GrB_Matrix  dp          =  DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix  dm          =  DELTA_MATRIX_DELTA_MINUS(C);

#ifdef DELTA_DEBUG
	info = GxB_Matrix_type(&type, m);
	ASSERT(info == GrB_SUCCESS);
	ASSERT(type == GrB_BOOL);

	info = GrB_Matrix_extractElement(&dm_x, dm, i, j);
	ASSERT(info == GrB_NO_VALUE);
#endif

	if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(C)) {
		info = Delta_Matrix_removeElement_BOOL(C->transposed, j, i);
		if(info != GrB_SUCCESS) {
			return info;
		}
	}

	//--------------------------------------------------------------------------
	// entry exists in 'M'
	//--------------------------------------------------------------------------

	info = GrB_Matrix_extractElement(&m_x, m, i, j);
	in_m = (info == GrB_SUCCESS);

	if(in_m) {
		// mark deletion in delta minus
		info = GrB_Matrix_setElement(m, false, i, j);
		info = GrB_Matrix_setElement(dm, true, i, j);
		ASSERT(info == GrB_SUCCESS);
		Delta_Matrix_setDirty(C);
		return info;
	}

	//--------------------------------------------------------------------------
	// entry exists in 'delta-plus'
	//--------------------------------------------------------------------------


	// remove entry from 'dp'
	info = GrB_Matrix_removeElement(dp, i, j);
	ASSERT(info == GrB_SUCCESS);
	Delta_Matrix_setDirty(C);
	return info;
}

GrB_Info Delta_Matrix_removeElement_UINT64
(
    Delta_Matrix C,                    // matrix to remove entry from
    GrB_Index i,                    // row index
    GrB_Index j                     // column index
) {
	ASSERT(C);
	Delta_Matrix_checkBounds(C, i, j);

	uint64_t    m_x;
	uint64_t    dm_x;
	uint64_t    dp_x;
	GrB_Info    info;
	GrB_Type    type;
	bool        in_m        =  false;
	bool        in_dp       =  false;
	bool        in_dm       =  false;
	GrB_Matrix  m           =  DELTA_MATRIX_M(C);
	GrB_Matrix  dp          =  DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix  dm          =  DELTA_MATRIX_DELTA_MINUS(C);

#ifdef DELTA_DEBUG
	info = GxB_Matrix_type(&type, m);
	ASSERT(info == GrB_SUCCESS);
	ASSERT(type == GrB_UINT64);
#endif

	if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(C)) {
		info = Delta_Matrix_removeElement_BOOL(C->transposed, j, i);
		if(info != GrB_SUCCESS) {
			return info;
		}
	}

	info = GrB_Matrix_extractElement(&m_x, m, i, j);
	in_m = (info == GrB_SUCCESS);

	info = GrB_Matrix_extractElement(&dp_x, dp, i, j);
	in_dp = (info == GrB_SUCCESS);

	info = GrB_Matrix_extractElement(&dm_x, dm, i, j);
	in_dm = (info == GrB_SUCCESS);

	// mask 'in_m' incase it is marked for deletion
	in_m = in_m && !(in_dm);

	// entry missing from both 'm' and 'dp'
	if(!(in_m || in_dp)) {
		return GrB_NO_VALUE;
	}

	// entry can't exists in both 'm' and 'dp'
	ASSERT(in_m != in_dp);

	//--------------------------------------------------------------------------
	// entry exists in 'M'
	//--------------------------------------------------------------------------

	if(in_m) {
		// mark deletion in M
		info = GrB_Matrix_setElement_UINT64(m, MSB_MASK, i, j);
		ASSERT(info == GrB_SUCCESS);
		// mark deletion in delta minus
		info = GrB_Matrix_setElement(dm, true, i, j);
		ASSERT(info == GrB_SUCCESS);
	}

	//--------------------------------------------------------------------------
	// entry exists in 'delta-plus'
	//--------------------------------------------------------------------------

	if(in_dp) {
		// remove entry from 'dp'
		info = GrB_Matrix_removeElement(dp, i, j);
		ASSERT(info == GrB_SUCCESS);
	}

	Delta_Matrix_setDirty(C);
	return info;
}

GrB_Info Delta_Matrix_removeElements
(
	Delta_Matrix C,  // matrix to remove entry from
	GrB_Matrix A     // matrix filled with elements to remove
) {
	ASSERT(C != NULL);
	ASSERT(A != NULL);
	ASSERT(!DELTA_MATRIX_MAINTAIN_TRANSPOSE(C));
	GrB_Info    info;
	GrB_Matrix  m  =  DELTA_MATRIX_M(C);
	GrB_Matrix  dp =  DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix  dm =  DELTA_MATRIX_DELTA_MINUS(C);

	// add edges in m and A to dm
	info = GrB_Matrix_eWiseMult_BinaryOp(
		dm, NULL, GrB_ONEB_BOOL, GrB_ONEB_BOOL, m, A, NULL) ;	
	ASSERT(info == GrB_SUCCESS);

	// remove edges in dp that are also in A.
	info = GrB_transpose(dp, A, NULL, dp, GrB_DESC_RSCT0);
	ASSERT(info == GrB_SUCCESS);

	// x XOR (x AND TRUE) - Always outputs false.
	// Done like this in the hope that GBLAS will recognize this can be done inplace.
	info = GrB_eWiseMult(m, NULL, GrB_LXOR, GrB_LAND, m, A, NULL);
	ASSERT(info == GrB_SUCCESS);

	Delta_Matrix_validate(C);

	Delta_Matrix_setDirty(C);
	return info;
}


