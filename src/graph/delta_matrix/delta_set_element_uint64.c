/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "GraphBLAS.h"
#include "RG.h"
#include "delta_utils.h"
#include "delta_matrix.h"
#include "../../util/arr.h"

#define DM_setElement                                                          \
	Delta_Matrix_checkBounds(C, i, j);                                         \
                                                                               \
	GrB_Info info;                                                             \
	bool entry_exists      = false;  /*  M[i,j] exists */                      \
	bool mark_for_deletion = false;  /*  dm[i,j] exists */                     \
                                                                               \
	if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(C)) {                                   \
		info =  Delta_Matrix_setElement_BOOL(C->transposed, j, i);             \
		if(info != GrB_SUCCESS) {                                              \
			return info;                                                       \
		}                                                                      \
	}                                                                          \
                                                                               \
	GrB_Matrix m  = DELTA_MATRIX_M(C);                                         \
	GrB_Matrix dp = DELTA_MATRIX_DELTA_PLUS(C);                                \
	GrB_Matrix dm = DELTA_MATRIX_DELTA_MINUS(C);                               \
                                                                               \
	/*-------------------------------------------------------------------------\
		check deleted                                                          \
	-------------------------------------------------------------------------*/\
                                                                               \
	GrB_OK (info = GxB_Matrix_isStoredElement(dm, i, j));                      \
	mark_for_deletion = (info == GrB_SUCCESS);                                 \
                                                                               \
	if(mark_for_deletion) { /* m contains single edge, simple replace */       \
		/* clear dm[i,j] */                                                    \
		GrB_OK (GrB_Matrix_removeElement(dm, i, j));                           \
                                                                               \
		/* overwrite m[i,j] */                                                 \
		GrB_OK (GrB_Matrix_setElement(m, x, i, j));                            \
	} else {                                                                   \
		/* entry isn't marked for deletion                                     \
		   see if entry already exists in 'm'                                  \
		   we'll prefer setting entry in 'm' incase it already exists          \
		   otherwise we'll set the entry in 'delta-plus' */                    \
		GrB_OK (info = GxB_Matrix_isStoredElement(m, i, j));                   \
		entry_exists = (info == GrB_SUCCESS);                                  \
                                                                               \
		if(entry_exists) {                                                     \
			/* update entry at m[i,j] */                                       \
			info = GrB_Matrix_setElement(m, x, i, j);                          \
		} else {                                                               \
			/* update entry at dp[i,j] */                                      \
			info = GrB_Matrix_setElement(dp, x, i, j);                         \
		}                                                                      \
	}                                                                          \
                                                                               \
	Delta_Matrix_setDirty(C);                                                  \
	return info;

// C (i,j) = x
GrB_Info Delta_Matrix_setElement_UINT64
(
    Delta_Matrix C,  // matrix to modify
    uint64_t x,      // scalar to assign to C(i,j)
    GrB_Index i,     // row index
    GrB_Index j      // column index
) {
	// validate
	ASSERT(C);
	GrB_Type ty = NULL;
	GrB_OK(GxB_Matrix_type(&ty, DELTA_MATRIX_M(C)));
	ASSERT(ty == GrB_UINT64);

	// This macro contains the full function definition which does not change
	// with the type. This is because the GraphBLAS generic macro will chose the
	// right function given the C type of x.
	DM_setElement
}

// C (i,j) = x
GrB_Info Delta_Matrix_setElement_UINT16
(
    Delta_Matrix C,  // matrix to modify
    uint16_t x,      // scalar to assign to C(i,j)
    GrB_Index i,     // row index
    GrB_Index j      // column index
) {
	// validate
	ASSERT(C);
	GrB_Type ty = NULL;
	GrB_OK(GxB_Matrix_type(&ty, DELTA_MATRIX_M(C)));
	ASSERT(ty == GrB_UINT16);

	// This macro contains the full function definition which does not change
	// with the type. This is because the GraphBLAS generic macro will chose the
	// right function given the C type of x.
	DM_setElement
}


