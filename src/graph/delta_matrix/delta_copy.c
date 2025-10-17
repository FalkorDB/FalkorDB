/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_utils.h"
#include "delta_matrix.h"
#include "../../util/rmalloc.h"

// copy matrix A to matrix C
// does not set the transpose
GrB_Info Delta_Matrix_dup
(
	Delta_Matrix *C,      // output matrix
	const Delta_Matrix A  // input matrix
) {
	ASSERT(C != NULL);
	ASSERT(A != NULL);
	
	Delta_Matrix _C    = rm_calloc(sizeof(_Delta_Matrix), 1);

	GrB_Matrix   in_m  = DELTA_MATRIX_M(A);
	GrB_Matrix   in_dp = DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix   in_dm = DELTA_MATRIX_DELTA_MINUS(A);


	GrB_OK (GrB_Matrix_dup(&DELTA_MATRIX_M(_C), in_m));
	GrB_OK (GrB_Matrix_dup(&DELTA_MATRIX_DELTA_PLUS(_C), in_dp));
	GrB_OK (GrB_Matrix_dup(&DELTA_MATRIX_DELTA_MINUS(_C), in_dm));

	int mutex_res = pthread_mutex_init(&_C->mutex, NULL);
	ASSERT(mutex_res == 0);

	Delta_Matrix_validate(_C, true);
	*C = _C;
	return GrB_SUCCESS;
}

