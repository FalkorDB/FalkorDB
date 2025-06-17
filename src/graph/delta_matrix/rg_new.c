/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
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
	GrB_Info info;
	A->dirty    =  false;

	//--------------------------------------------------------------------------
	// create m, delta-plus and delta-minus
	//--------------------------------------------------------------------------

	//--------------------------------------------------------------------------
	// m, can be either hypersparse or sparse
	//--------------------------------------------------------------------------
	info = GrB_Matrix_new(&A->matrix, type, nrows, ncols);
	ASSERT(info == GrB_SUCCESS);
	info = GxB_set(A->matrix, GxB_SPARSITY_CONTROL, GxB_SPARSE | GxB_HYPERSPARSE);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// delta-plus, always hypersparse
	//--------------------------------------------------------------------------
	info = GrB_Matrix_new(&A->delta_plus, type, nrows, ncols);
	ASSERT(info == GrB_SUCCESS);
	info = GxB_set(A->delta_plus, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE);
	ASSERT(info == GrB_SUCCESS);
	info = GxB_set(A->delta_plus, GxB_HYPER_SWITCH, GxB_ALWAYS_HYPER);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// delta-minus, always hypersparse
	//--------------------------------------------------------------------------
	info = GrB_Matrix_new(&A->delta_minus, GrB_BOOL, nrows, ncols);
	ASSERT(info == GrB_SUCCESS);
	info = GxB_set(A->delta_minus, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE);
	ASSERT(info == GrB_SUCCESS);
	info = GxB_set(A->delta_minus, GxB_HYPER_SWITCH, GxB_ALWAYS_HYPER);
	ASSERT(info == GrB_SUCCESS);

	return info;
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
	GrB_Info info;
	Delta_Matrix matrix = rm_calloc(1, sizeof(_Delta_Matrix));

	//--------------------------------------------------------------------------
	// input validations
	//--------------------------------------------------------------------------

	// supported types: boolean and uint64
	ASSERT(type == GrB_BOOL || type == GrB_UINT64);

	info = _Delta_Matrix_init(matrix, type, nrows, ncols);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// create transpose matrix if required
	//--------------------------------------------------------------------------

	if(transpose) {
		matrix->transposed = rm_calloc(1, sizeof(_Delta_Matrix));
		info = _Delta_Matrix_init(matrix->transposed, GrB_BOOL, ncols, nrows);
		ASSERT(info == GrB_SUCCESS);
	}

	int mutex_res = pthread_mutex_init(&matrix->mutex, NULL);
	ASSERT(mutex_res == 0);

	*A = matrix;
	return info;
}

