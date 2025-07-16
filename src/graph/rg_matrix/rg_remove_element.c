/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "rg_matrix.h"
#include "rg_utils.h"
#include "../../util/arr.h"
#include "../../util/rmalloc.h"

GrB_Info RG_Matrix_removeElement_BOOL
(
	RG_Matrix C,                    // matrix to remove entry from
	GrB_Index i,                    // row index
	GrB_Index j                     // column index
) {
	ASSERT(C);
	RG_Matrix_checkBounds(C, i, j);

	bool        m_x;
	bool        dm_x;
	bool        dp_x;
	GrB_Info    info;
	GrB_Type    type;
	bool        in_m        =  false;
	bool        in_dp       =  false;
	bool        in_dm       =  false;
	GrB_Matrix  m           =  RG_MATRIX_M(C);
	GrB_Matrix  dp          =  RG_MATRIX_DELTA_PLUS(C);
	GrB_Matrix  dm          =  RG_MATRIX_DELTA_MINUS(C);

#ifdef RG_DEBUG
	info = GxB_Matrix_type(&type, m);
	ASSERT(info == GrB_SUCCESS);
	ASSERT(type == GrB_BOOL);

	info = GrB_Matrix_extractElement(&dm_x, dm, i, j);
	ASSERT(info == GrB_NO_VALUE);
#endif

	if(RG_MATRIX_MAINTAIN_TRANSPOSE(C)) {
		info = RG_Matrix_removeElement_BOOL(C->transposed, j, i);
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
		info = GrB_Matrix_setElement(dm, true, i, j);
		ASSERT(info == GrB_SUCCESS);
		RG_Matrix_setDirty(C);
		return info;
	}

	//--------------------------------------------------------------------------
	// entry exists in 'delta-plus'
	//--------------------------------------------------------------------------


	// remove entry from 'dp'
	info = GrB_Matrix_removeElement(dp, i, j);
	ASSERT(info == GrB_SUCCESS);
	RG_Matrix_setDirty(C);
	return info;
}

GrB_Info RG_Matrix_removeElement_UINT64
(
    RG_Matrix C,                    // matrix to remove entry from
    GrB_Index i,                    // row index
    GrB_Index j                     // column index
) {
	ASSERT(C);
	RG_Matrix_checkBounds(C, i, j);

	uint64_t    m_x;
	uint64_t    dm_x;
	uint64_t    dp_x;
	GrB_Info    info;
	GrB_Type    type;
	bool        in_m        =  false;
	bool        in_dp       =  false;
	bool        in_dm       =  false;
	GrB_Matrix  m           =  RG_MATRIX_M(C);
	GrB_Matrix  dp          =  RG_MATRIX_DELTA_PLUS(C);
	GrB_Matrix  dm          =  RG_MATRIX_DELTA_MINUS(C);

#ifdef RG_DEBUG
	info = GxB_Matrix_type(&type, m);
	ASSERT(info == GrB_SUCCESS);
	ASSERT(type == GrB_UINT64);
#endif

	if(RG_MATRIX_MAINTAIN_TRANSPOSE(C)) {
		info = RG_Matrix_removeElement_BOOL(C->transposed, j, i);
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
		// free multi-edge entry, leave M[i,j] dirty
		if((SINGLE_EDGE(m_x)) == false) {
			m_x = CLEAR_MSB(m_x);
			array_free((uint64_t *)m_x);
		}

		// mark deletion in delta minus
		info = GrB_Matrix_setElement(dm, true, i, j);
		ASSERT(info == GrB_SUCCESS);
	}

	//--------------------------------------------------------------------------
	// entry exists in 'delta-plus'
	//--------------------------------------------------------------------------

	if(in_dp) {
		// free multi-edge entry
		if((SINGLE_EDGE(dp_x)) == false) {
			dp_x = CLEAR_MSB(dp_x);
			array_free((uint64_t *)dp_x);
		}

		// remove entry from 'dp'
		info = GrB_Matrix_removeElement(dp, i, j);
		ASSERT(info == GrB_SUCCESS);
	}

	RG_Matrix_setDirty(C);
	return info;
}

