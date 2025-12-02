/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"
#include "../../util/rmalloc.h"

static GrB_Info _Delta_Matrix_init
(
	Delta_Matrix A,
	GrB_Type type,
	GrB_Index nrows,
	GrB_Index ncols
) {
	A->dirty = false;

	//--------------------------------------------------------------------------
	// create m, delta-plus and delta-minus
	//--------------------------------------------------------------------------

	//--------------------------------------------------------------------------
	// m, can be either hypersparse or sparse
	//--------------------------------------------------------------------------
	GrB_OK(GrB_Matrix_new(&A->matrix, type, nrows, ncols));
	GrB_OK(GxB_set(A->matrix, GxB_SPARSITY_CONTROL, GxB_SPARSE | GxB_HYPERSPARSE));

	//--------------------------------------------------------------------------
	// delta-plus, always hypersparse
	//--------------------------------------------------------------------------
	GrB_OK(GrB_Matrix_new(&A->delta_plus, type, nrows, ncols));
	GrB_OK(GxB_set(A->delta_plus, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE));
	GrB_OK(GxB_set(A->delta_plus, GxB_HYPER_SWITCH, GxB_ALWAYS_HYPER));

	//--------------------------------------------------------------------------
	// delta-minus, always hypersparse
	//--------------------------------------------------------------------------
	GrB_OK(GrB_Matrix_new(&A->delta_minus, GrB_BOOL, nrows, ncols));
	GrB_OK(GxB_set(A->delta_minus, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE));
	GrB_OK(GxB_set(A->delta_minus, GxB_HYPER_SWITCH, GxB_ALWAYS_HYPER));

	return GrB_SUCCESS;
}

// creates a new matrix
GrB_Info Delta_Matrix_new
(
	Delta_Matrix *A,
	GrB_Type type,
	GrB_Index nrows,
	GrB_Index ncols,
	bool transpose
) {
	Delta_Matrix matrix = rm_calloc(1, sizeof(_Delta_Matrix));
	//--------------------------------------------------------------------------
	// input validations
	//--------------------------------------------------------------------------

	// supported types: boolean and uint64 and uint16
	ASSERT(type == GrB_BOOL || type == GrB_UINT16 || type == GrB_UINT64);

	GrB_OK(_Delta_Matrix_init(matrix, type, nrows, ncols));

	//--------------------------------------------------------------------------
	// create transpose matrix if required
	//--------------------------------------------------------------------------

	if(transpose) {
		matrix->transposed = rm_calloc(1, sizeof(_Delta_Matrix));
		GrB_OK(_Delta_Matrix_init(matrix->transposed, GrB_BOOL, ncols, nrows));
	}

	int mutex_res = pthread_mutex_init(&matrix->mutex, NULL);
	ASSERT(mutex_res == 0);

	*A = matrix;
	return GrB_SUCCESS;
}

