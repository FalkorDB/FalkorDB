/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "GraphBLAS.h"
#include "delta_utils.h"
#include "delta_matrix.h"

// C (i,j) = accum(C(i,j), x)
#define DM_assign_scalar(TYPE_SUFFIX, CTYPE)                                   \
GrB_Info Delta_Matrix_assign_scalar_##TYPE_SUFFIX                              \
(                                                                              \
	Delta_Matrix C,            /* input/output matrix for results      */      \
	const GrB_BinaryOp accum,  /* optional accum for Z=accum(C(I,J),x) */      \
	CTYPE x,                   /* scalar to assign to C(i,j)           */      \
	GrB_Index i,               /* row index                            */      \
	GrB_Index j                /* column index                         */      \
) {                                                                            \
	/* validate */                                                             \
	ASSERT (C != NULL) ;                                                       \
	Delta_Matrix_checkBounds (C, i, j) ;                                       \
                                                                               \
	uint64_t v ;                                                               \
	GrB_Info info ;                                                            \
	bool     in_M  = false ;  /* M[i,j] exists  */                             \
	bool     in_DM = false ;  /* dm[i,j] exists */                             \
                                                                               \
	if (DELTA_MATRIX_MAINTAIN_TRANSPOSE (C)) {                                 \
		info = Delta_Matrix_setElement_BOOL (C->transposed, true, j, i) ;      \
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
	/* check in m */                                                           \
	info = GxB_Matrix_isStoredElement (m, i, j) ;                              \
	in_M = (info == GrB_SUCCESS) ;                                             \
                                                                               \
	if (in_M) {                                                                \
		/* NOTE: Checking if elements is deleted will cause significant slow */\
		/* down due to read after write. Therefore, it is not checked. So, we*/\
		/* assume that the "zombie" value in m is the identity of accum.     */\
		GrB_OK (GrB_assign (m, NULL, accum, x, &i, 1, &j, 1, NULL)) ;          \
		GrB_OK (GrB_Matrix_removeElement (dm, i, j)) ;                         \
	} else {                                                                   \
		/* update entry at dp[i,j] */                                          \
		GrB_OK (GrB_assign (dp, NULL, accum, x, &i, 1, &j, 1, NULL)) ;         \
	}                                                                          \
                                                                               \
	Delta_Matrix_setDirty (C) ;                                                \
	return GrB_SUCCESS ;                                                       \
}

//------------------------------------------------------------------------------
// Function definitions (contained entirely within the following macros)
//------------------------------------------------------------------------------

DM_assign_scalar(UINT64, uint64_t)
DM_assign_scalar(UINT16, uint16_t)

