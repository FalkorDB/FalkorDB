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
#include "../../util/arr.h"
#include "../../util/rmalloc.h"

static bool _removeEntryFromMultiValArr
(
	uint64_t **entries,  // multi-value array
	uint64_t entry       // element to remove output new value
) {
	ASSERT(*entries != NULL);

	uint  i  =  0;
	uint  n  =  array_len(*entries);

	// search for entry
	for(; i < n; i++) {
		if((*entries)[i] == entry) {
			break;
		}
	}

	ASSERT(i < n);

	// remove located entry
	// migrate last element and reduce array size
	array_del_fast(*entries, i);

	// incase we're left with a single entry revert back to scalar
	if(array_len(*entries) == 1) {
		entry = (*entries)[0];
		array_free(*entries);
		*entries = (uint64_t *)entry;
		return true;
	}

	return false;
}

static GrB_Info _removeElementMultiVal
(
	GrB_Matrix A,                   // matrix to remove entry from
	GrB_Index i,                    // row index
	GrB_Index j,                    // column index
	uint64_t  v                     // value to remove
) {
	ASSERT(A);

	uint64_t  x;
	uint64_t  tx;
	GrB_Info  info;

	info = GrB_Matrix_extractElement(&x, A, i, j);
	ASSERT(info == GrB_SUCCESS);
	ASSERT((SINGLE_EDGE(x)) == false);

	// remove entry from multi-value
	x  = CLEAR_MSB(x);
	uint64_t *entries  = (uint64_t *)x;
	if(_removeEntryFromMultiValArr(&entries, v)) {
		// update entry
		x = (uint64_t)entries;
		info = GrB_Matrix_setElement(A, x, i, j);
	}

	return info;
}

GrB_Info RG_Matrix_removeEntry_UINT64
(
	RG_Matrix C,                    // matrix to remove entry from
	GrB_Index i,                    // row index
	GrB_Index j,                    // column index
	uint64_t  v,                    // value to remove
	bool     *entry_deleted         // is entry deleted
) {
	ASSERT(C);
	ASSERT(entry_deleted != NULL);
	RG_Matrix_checkBounds(C, i, j);

	uint64_t    m_x;
	uint64_t    dp_x;
	GrB_Info    info;
	GrB_Type    type;
	bool        in_m        =  false;
	GrB_Matrix  m           =  RG_MATRIX_M(C);
	GrB_Matrix  dp          =  RG_MATRIX_DELTA_PLUS(C);
	GrB_Matrix  dm          =  RG_MATRIX_DELTA_MINUS(C);

	*entry_deleted = false;
	

#ifdef RG_DEBUG
	info = GxB_Matrix_type(&type, m);
	ASSERT(info == GrB_SUCCESS);
	ASSERT(type == GrB_UINT64);
	
	bool dm_x;
	info = GrB_Matrix_extractElement(&dm_x, dm, i, j);
	ASSERT(info == GrB_NO_VALUE);
#endif

	// entry should exists in either delta-plus or main
	// locate entry
	info = GrB_Matrix_extractElement(&m_x, m, i, j);
	in_m = (info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// entry exists in 'M'
	//--------------------------------------------------------------------------

	if(in_m) {
		if(SINGLE_EDGE(m_x)) {
			*entry_deleted = true;
			// mark deletion in delta minus
			info = GrB_Matrix_setElement(dm, true, i, j);
			ASSERT(info == GrB_SUCCESS);
			info = RG_Matrix_removeElement_BOOL(C->transposed, j, i);
			ASSERT(info == GrB_SUCCESS)
			RG_Matrix_setDirty(C);
		} else {
			info = _removeElementMultiVal(m, i, j, v);
			ASSERT(info == GrB_SUCCESS);
		}
		return info;
	}

	//--------------------------------------------------------------------------
	// entry exists in 'delta-plus'
	//--------------------------------------------------------------------------

	info = GrB_Matrix_extractElement(&dp_x, dp, i, j);
	if(info != GrB_SUCCESS) return info;

	if(SINGLE_EDGE(dp_x)) {
		*entry_deleted = true;
		info = GrB_Matrix_removeElement(dp, i, j);
		ASSERT(info == GrB_SUCCESS);
		info = RG_Matrix_removeElement_BOOL(C->transposed, j, i);
		ASSERT(info == GrB_SUCCESS)
		RG_Matrix_setDirty(C);
	} else {
		info = _removeElementMultiVal(dp, i, j, v);
		ASSERT(info == GrB_SUCCESS);
	}
	return info;
}
