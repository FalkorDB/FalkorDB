/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "GraphBLAS.h"
#include "RG.h"
#include "delta_utils.h"
#include "delta_matrix.h"

#define DM_setElement(TYPE_SUFFIX, CTYPE)                                      \
GrB_Info Delta_Matrix_setElement_##TYPE_SUFFIX                                 \
(                                                                              \
    Delta_Matrix C,  /* matrix to modify           */                          \
    CTYPE x,         /* scalar to assign to C(i,j) */                          \
    GrB_Index i,     /* row index                  */                          \
    GrB_Index j      /* column index               */                          \
) {                                                                            \
	/* validate */                                                             \
	ASSERT (C) ;                                                               \
	Delta_Matrix_checkBounds (C, i, j) ;                                       \
                                                                               \
	GrB_Info info ;                                                            \
	bool in_M  = false ;  /*  M[i,j] exists */                                 \
                                                                               \
	if (DELTA_MATRIX_MAINTAIN_TRANSPOSE (C)) {                                 \
		info =  Delta_Matrix_setElement_BOOL (C->transposed, true, j, i) ;     \
		if (info != GrB_SUCCESS) {                                             \
			return info ;                                                      \
		}                                                                      \
	}                                                                          \
                                                                               \
	GrB_Matrix m  = DELTA_MATRIX_M (C) ;                                       \
	GrB_Matrix dp = DELTA_MATRIX_DELTA_PLUS (C) ;                              \
	GrB_Matrix dm = DELTA_MATRIX_DELTA_MINUS (C) ;                             \
                                                                               \
	GRB_MATRIX_TYPE_ASSERT(m, GrB_##TYPE_SUFFIX) ;                             \
                                                                               \
	/* check if entry exists in m */                                           \
	GrB_OK (info = GxB_Matrix_isStoredElement (m, i, j)) ;                     \
	in_M = (info == GrB_SUCCESS) ;                                             \
                                                                               \
	if (in_M) {                                                                \
		/* clear dm[i,j] will remove if it exists, and do nothing otherwise */ \
		GrB_OK (GrB_Matrix_removeElement (dm, i, j)) ;                         \
                                                                               \
		/* overwrite m[i,j] */                                                 \
		GrB_OK (GrB_Matrix_setElement (m, x, i, j)) ;                          \
	} else {                                                                   \
		/* update entry at dp[i,j] */                                          \
		GrB_OK (GrB_Matrix_setElement (dp, x, i, j)) ;                         \
	}                                                                          \
                                                                               \
	Delta_Matrix_setDirty (C) ;                                                \
	return GrB_SUCCESS ;                                                       \
}

//------------------------------------------------------------------------------
// Function definitions (contained entirely within the following macros)
//------------------------------------------------------------------------------

DM_setElement(BOOL, bool)
DM_setElement(UINT16, uint16_t)
DM_setElement(UINT64, uint64_t)

