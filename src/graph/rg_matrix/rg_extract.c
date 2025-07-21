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

GrB_Info RG_Matrix_extractElement_BOOL     // x = A(i,j)
(
    bool *x,                               // extracted scalar
    const RG_Matrix A,                     // matrix to extract a scalar from
    GrB_Index i,                           // row index
    GrB_Index j                            // column index
) {
	ASSERT(x != NULL);
	ASSERT(A != NULL);

	GrB_Info info;
	GrB_Matrix  m      =  RG_MATRIX_M(A);
	GrB_Matrix  dp     =  RG_MATRIX_DELTA_PLUS(A);
	GrB_Matrix  dm     =  RG_MATRIX_DELTA_MINUS(A);

	// if 'delta-plus' exists return dp[i,j]
	info = GrB_Matrix_extractElement(x, dp, i, j);
	if(info == GrB_SUCCESS) {
		return info;
	}

	// if dm[i,j] exists, return no value
	info = GrB_Matrix_extractElement(x, dm, i, j);
	if(info == GrB_SUCCESS) {
		// entry marked for deletion
		return GrB_NO_VALUE;
	}

	// entry isn't marked for deletion, see if it exists in 'm'
	info = GrB_Matrix_extractElement(x, m, i, j);
	return info;
}

GrB_Info RG_Matrix_extractElement_UINT64   // x = A(i,j)
(
    uint64_t *x,                           // extracted scalar
    const RG_Matrix A,                     // matrix to extract a scalar from
    GrB_Index i,                           // row index
    GrB_Index j                            // column index
) {
	ASSERT(x != NULL);
	ASSERT(A != NULL);

	GrB_Info info;
	GrB_Matrix  m      =  RG_MATRIX_M(A);
	GrB_Matrix  dp     =  RG_MATRIX_DELTA_PLUS(A);
	GrB_Matrix  dm     =  RG_MATRIX_DELTA_MINUS(A);

	// if dp[i,j] exists return it
	info = GrB_Matrix_extractElement(x, dp, i, j);
	if(info == GrB_SUCCESS) {
		return info;
	}

	// if dm[i,j] exists, return no value
	info = GrB_Matrix_extractElement(x, dm, i, j);
	if(info == GrB_SUCCESS) {
		// entry marked for deletion
		return GrB_NO_VALUE;
	}

	// entry isn't marked for deletion, see if it exists in 'm'
	info = GrB_Matrix_extractElement(x, m, i, j);
	return info;
}

