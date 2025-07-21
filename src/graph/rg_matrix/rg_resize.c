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

GrB_Info RG_Matrix_resize       // change the size of a matrix
(
    RG_Matrix C,                // matrix to modify
    GrB_Index nrows_new,        // new number of rows in matrix
    GrB_Index ncols_new         // new number of columns in matrix
) {
	ASSERT(C != NULL);
	GrB_Info info;

	if(RG_MATRIX_MAINTAIN_TRANSPOSE(C)) {
		info = RG_Matrix_resize(C->transposed, ncols_new, nrows_new);
		ASSERT(info == GrB_SUCCESS);
	}

	GrB_Matrix  m            =  RG_MATRIX_M(C);
	GrB_Matrix  delta_plus   =  RG_MATRIX_DELTA_PLUS(C);
	GrB_Matrix  delta_minus  =  RG_MATRIX_DELTA_MINUS(C);

	info = GrB_Matrix_resize(m, nrows_new, ncols_new);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_resize(delta_plus, nrows_new, ncols_new);
	ASSERT(info == GrB_SUCCESS);
	
	info = GrB_Matrix_resize(delta_minus, nrows_new, ncols_new);
	ASSERT(info == GrB_SUCCESS);
	
	return info;
}

