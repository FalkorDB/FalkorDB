/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "GraphBLAS.h"
#include "RG.h"
#include "delta_utils.h"
#include "delta_matrix.h"

#define DM_assign_scalar                                                       \
{                                                                              \
	ASSERT (C != NULL) ;                                                       \
	Delta_Matrix_checkBounds (C, i, j) ;                                       \
                                                                               \
	uint64_t v ;                                                               \
	GrB_Info info ;                                                            \
	bool     entry_exists      = false ;  /* M[i,j] exists  */                 \
	bool     mark_for_deletion = false ;  /* dm[i,j] exists */                 \
                                                                               \
	if (DELTA_MATRIX_MAINTAIN_TRANSPOSE (C)) {                                 \
		info = Delta_Matrix_setElement_BOOL (C->transposed, j, i) ;            \
		if (info != GrB_SUCCESS) {                                             \
			return info ;                                                      \
		}                                                                      \
	}                                                                          \
                                                                               \
	GrB_Matrix m  = DELTA_MATRIX_M (C) ;                                       \
	GrB_Matrix dp = DELTA_MATRIX_DELTA_PLUS (C) ;                              \
	GrB_Matrix dm = DELTA_MATRIX_DELTA_MINUS (C) ;                             \
                                                                               \
	/*-------------------------------------------------------------------------\
	// check deleted                                                           \
	//-----------------------------------------------------------------------*/\
                                                                               \
	info = GxB_Matrix_isStoredElement(dm, i, j) ;                              \
	mark_for_deletion = (info == GrB_SUCCESS) ;                                \
                                                                               \
	if (mark_for_deletion) { /* m contains single edge, simple replace */      \
		/* clear dm[i,j] */                                                    \
		GrB_OK (GrB_Matrix_removeElement (dm, i, j)) ;                         \
                                                                               \
		/* overwrite m[i,j] */                                                 \
		GrB_OK (GrB_Matrix_setElement (m, x, i, j)) ;                          \
	} else {                                                                   \
		/* entry isn't marked for deletion                                     \
		// see if entry already exists in 'm'                                  \
		// we'll prefer setting entry in 'm' incase it already exists          \
		// otherwise we'll set the entry in 'delta-plus' */                    \
		info = GxB_Matrix_isStoredElement(m, i, j) ;                           \
		entry_exists = (info == GrB_SUCCESS) ;                                 \
                                                                               \
		if(entry_exists) {                                                     \
			/* update entry at m[i,j] */                                       \
			GrB_OK (GrB_assign (m, NULL, accum, x, &i, 1, &j, 1,               \
				NULL)) ;                                                       \
		} else {                                                               \
			/* update entry at dp[i,j] */                                      \
			GrB_OK (GrB_assign (dp, NULL, accum, x, &i, 1, &j, 1,              \
				NULL)) ;                                                       \
		}                                                                      \
	}                                                                          \
                                                                               \
	Delta_Matrix_setDirty (C) ;                                                \
	return GrB_SUCCESS ;                                                       \
}

// C (i,j) = accum(C(i,j), x)
GrB_Info Delta_Matrix_assign_scalar_UINT64
(
    Delta_Matrix C,            // input/output matrix for results
    const GrB_BinaryOp accum,  // optional accum for Z=accum(C(I,J),x)
	uint64_t x,                // scalar to assign to C(i,j)
	GrB_Index i,               // row index
	GrB_Index j                // column index
)
DM_assign_scalar

// C (i,j) = accum(C(i,j), x)
GrB_Info Delta_Matrix_assign_scalar_UINT16
(
    Delta_Matrix C,            // input/output matrix for results
    const GrB_BinaryOp accum,  // optional accum for Z=accum(C(I,J),x)
	uint16_t x,                // scalar to assign to C(i,j)
	GrB_Index i,               // row index
	GrB_Index j                // column index
)
DM_assign_scalar
