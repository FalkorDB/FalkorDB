/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "GraphBLAS.h"
#include "RG.h"
#include "delta_matrix.h"
#include "graph/delta_matrix/delta_utils.h"

#define DM_extractElement(TYPE_SUFFIX, CTYPE)                                  \
GrB_Info Delta_Matrix_extractElement_##TYPE_SUFFIX                             \
(                                                                              \
	CTYPE *x,              /* extracted scalar                */               \
	const Delta_Matrix A,  /* matrix to extract a scalar from */               \
	GrB_Index i,           /* row index                       */               \
	GrB_Index j            /* column index                    */               \
) {                                                                            \
	/* validate */                                                             \
	ASSERT(x != NULL);                                                         \
	ASSERT(A != NULL);                                                         \
                                                                               \
	CTYPE      _x ;                                                            \
	GrB_Info   info;                                                           \
	GrB_Matrix m  = DELTA_MATRIX_M (A) ;                                       \
	GrB_Matrix dp = DELTA_MATRIX_DELTA_PLUS (A) ;                              \
	GrB_Matrix dm = DELTA_MATRIX_DELTA_MINUS (A) ;                             \
                                                                               \
	GRB_MATRIX_TYPE_ASSERT(m, GrB_##TYPE_SUFFIX)                               \
                                                                               \
	/* if dp[i,j] exists return it */                                          \
	GrB_OK (info = GrB_Matrix_extractElement (x, dp, i, j)) ;                  \
	if (info == GrB_SUCCESS) {                                                 \
		return GrB_SUCCESS ;                                                   \
	}                                                                          \
                                                                               \
	/* see if entry exists in 'm' */                                           \
	GrB_OK (info = GrB_Matrix_extractElement (&_x, m, i, j)) ;                 \
	if (info == GrB_NO_VALUE) {                                                \
		return GrB_NO_VALUE;                                                   \
	}                                                                          \
                                                                               \
	/* if dm[i,j] exists, return no value and do not set *x */                 \
	GrB_OK (info = GxB_Matrix_isStoredElement (dm, i, j)) ;                    \
	if (info == GrB_NO_VALUE) {                                                \
		*x = _x;                                                               \
	}                                                                          \
                                                                               \
	return (info == GrB_NO_VALUE) ? GrB_SUCCESS : GrB_NO_VALUE ;               \
}

//------------------------------------------------------------------------------
// Function definitions (contained entirely within the following macros)
//------------------------------------------------------------------------------
DM_extractElement(UINT64, uint64_t)
DM_extractElement(UINT16, uint16_t)

