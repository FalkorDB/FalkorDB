/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"
#define DM_extractElement                                                      \
GrB_Info info;                                                                 \
GrB_Matrix m  = DELTA_MATRIX_M(A);                                             \
GrB_Matrix dp = DELTA_MATRIX_DELTA_PLUS(A);                                    \
GrB_Matrix dm = DELTA_MATRIX_DELTA_MINUS(A);                                   \
                                                                               \
/* if dp[i,j] exists return it */                                              \
GrB_OK (info = GrB_Matrix_extractElement(x, dp, i, j));                        \
if(info == GrB_SUCCESS) {                                                      \
	return GrB_SUCCESS;                                                        \
}                                                                              \
                                                                               \
/* if dm[i,j] exists, return no value */                                       \
GrB_OK (info = GxB_Matrix_isStoredElement(dm, i, j));                          \
if(info == GrB_SUCCESS) {                                                      \
	/* entry marked for deletion */                                            \
	return GrB_NO_VALUE;                                                       \
}                                                                              \
                                                                               \
/* entry isn't marked for deletion, see if it exists in 'm' */                 \
GrB_OK (info = GrB_Matrix_extractElement(x, m, i, j));                         \
return info;


GrB_Info Delta_Matrix_extractElement_UINT64   // x = A(i,j)
(
    uint64_t *x,           // extracted scalar
    const Delta_Matrix A,  // matrix to extract a scalar from
    GrB_Index i,           // row index
    GrB_Index j            // column index
) {
	// validate
	ASSERT(x != NULL);
	ASSERT(A != NULL);
	GrB_Type ty = NULL;
	GrB_OK(GxB_Matrix_type(&ty, DELTA_MATRIX_M(A)));
	ASSERT(ty == GrB_UINT64);

	// This macro contains the full function definition which does not change
	// with the type. This is because the GraphBLAS generic macro will chose the
	// right function given the C type of x.
	DM_extractElement
}

GrB_Info Delta_Matrix_extractElement_UINT16   // x = A(i,j)
(
    uint16_t *x,           // extracted scalar
    const Delta_Matrix A,  // matrix to extract a scalar from
    GrB_Index i,           // row index
    GrB_Index j            // column index
) {
	// validate
	ASSERT(x != NULL);
	ASSERT(A != NULL);
	GrB_Type ty = NULL;
	GrB_OK(GxB_Matrix_type(&ty, DELTA_MATRIX_M(A)));
	ASSERT(ty == GrB_UINT16);

	// This macro contains the full function definition which does not change
	// with the type. This is because the GraphBLAS generic macro will chose the
	// right function given the C type of x.
	DM_extractElement
}
