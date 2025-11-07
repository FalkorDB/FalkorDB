/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_utils.h"
#include "delta_matrix.h"
#include "../../util/arr.h"

// C (i,j) += x
GrB_Info Delta_Matrix_Assign_Element_UINT64
(
	Delta_Matrix C,     // matrix to modify
	GrB_BinaryOp accum, // accumulator to apply to duplicates
	uint64_t x,         // scalar to assign to C(i,j)
	GrB_Index i,        // row index
	GrB_Index j         // column index
) {
	ASSERT(C != NULL);
	Delta_Matrix_checkBounds(C, i, j);

	uint64_t  v;
	GrB_Info  info;
	bool      entry_exists       =  false;          //  M[i,j] exists
	bool      mark_for_deletion  =  false;          //  dm[i,j] exists

	if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(C)) {
		info =  Delta_Matrix_setElement_BOOL(C->transposed, j, i);
		if(info != GrB_SUCCESS) {
			return info;
		}
	}

	GrB_Matrix m   = DELTA_MATRIX_M(C);
	GrB_Matrix dp  = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix dm  = DELTA_MATRIX_DELTA_MINUS(C);

#ifdef RG_DEBUG
	//--------------------------------------------------------------------------
	// validate type
	//--------------------------------------------------------------------------

	GrB_Type t;
	info = GxB_Matrix_type(&t, m);
	ASSERT(info == GrB_SUCCESS);
	ASSERT(t == GrB_UINT64);
#endif

	//--------------------------------------------------------------------------
	// check deleted
	//--------------------------------------------------------------------------

	GrB_OK (info = GrB_Matrix_extractElement(&v, dm, i, j));
	mark_for_deletion = (info == GrB_SUCCESS);

	if(mark_for_deletion) { // m contains single edge, simple replace
		// clear dm[i,j]
		GrB_OK (GrB_Matrix_removeElement(dm, i, j));

		// overwrite m[i,j]
		GrB_OK (GrB_Matrix_setElement_UINT64(m, x, i, j));
	} else {
		// entry isn't marked for deletion
		// see if entry already exists in 'm'
		// we'll prefer setting entry in 'm' incase it already exists
		// otherwise we'll set the entry in 'delta-plus'
		GrB_OK (info = GrB_Matrix_extractElement_UINT64(&v, m, i, j));
		entry_exists = (info == GrB_SUCCESS);

		if(entry_exists) {
			// update entry at m[i,j]
			GrB_OK (GrB_Matrix_assign_UINT64(m, NULL, accum, x, &i, 1, &j, 1, 
				NULL));
		} else {
			// update entry at dp[i,j]
			GrB_OK (GrB_Matrix_assign_UINT64(dp, NULL, accum, x, &i, 1, &j, 1, 
				NULL));
		}
	}

	Delta_Matrix_setDirty(C);
	return GrB_SUCCESS;
}

