/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"

// x = A(i,j)
GrB_Info Delta_Matrix_extractElement_BOOL     
(
    bool *x,               // extracted scalar
    const Delta_Matrix A,  // matrix to extract a scalar from
    GrB_Index i,           // row index
    GrB_Index j            // column index
) {
	ASSERT(A != NULL);

	GrB_Info info;
	GrB_Matrix  m  =  DELTA_MATRIX_M(A);
	GrB_Matrix  dp =  DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix  dm =  DELTA_MATRIX_DELTA_MINUS(A);
	bool        _x =  false;
	// if 'delta-plus' exists return dp[i,j]
	info = GrB_Matrix_extractElement(&_x, dp, i, j);
	if(info == GrB_SUCCESS) {
		if(x) *x = _x;
		return info;
	}

	// if dm[i,j] exists, return no value
	info = GxB_Matrix_isStoredElement(dm, i, j);
	if(info == GrB_SUCCESS) {
		// entry marked for deletion
		return GrB_NO_VALUE;
	}

	// entry isn't marked for deletion, see if it exists in 'm'
	info = GrB_Matrix_extractElement(&_x, m, i, j);
	if(x) *x = _x;
	return info;
}

GrB_Info Delta_Matrix_extractElement_UINT64   // x = A(i,j)
(
    uint64_t *x,                           // extracted scalar
    const Delta_Matrix A,                     // matrix to extract a scalar from
    GrB_Index i,                           // row index
    GrB_Index j                            // column index
) {
	ASSERT(A != NULL);

	GrB_Info info;
	GrB_Matrix  m      =  DELTA_MATRIX_M(A);
	GrB_Matrix  dp     =  DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix  dm     =  DELTA_MATRIX_DELTA_MINUS(A);
	uint64_t    _x     =  0;

	// if dp[i,j] exists return it
	info = GrB_Matrix_extractElement(&_x, dp, i, j);
	if(info == GrB_SUCCESS) {
		if(x) *x = _x;
		return info;
	}

	// if dm[i,j] exists, return no value
	info = GxB_Matrix_isStoredElement(dm, i, j);
	if(info == GrB_SUCCESS) {
		// entry marked for deletion
		return GrB_NO_VALUE;
	}

	// entry isn't marked for deletion, see if it exists in 'm'
	info = GrB_Matrix_extractElement(&_x, m, i, j);
	
	if(x) *x = _x;
	return info;
}

