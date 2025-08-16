/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"

// change the size of a matrix
GrB_Info Delta_Matrix_resize    
(
    Delta_Matrix C,       // matrix to modify
    GrB_Index nrows_new,  // new number of rows in matrix
    GrB_Index ncols_new   // new number of columns in matrix
) {
	ASSERT(C != NULL);

	if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(C)) {
		GrB_OK(Delta_Matrix_resize(C->transposed, ncols_new, nrows_new));
	}

	GrB_Matrix m           = DELTA_MATRIX_M(C);
	GrB_Matrix delta_plus  = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix delta_minus = DELTA_MATRIX_DELTA_MINUS(C);

	GrB_OK(GrB_Matrix_resize(m, nrows_new, ncols_new));
	GrB_OK(GrB_Matrix_resize(delta_plus, nrows_new, ncols_new));
	GrB_OK(GrB_Matrix_resize(delta_minus, nrows_new, ncols_new));
	
	return GrB_SUCCESS;
}

