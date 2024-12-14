/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "degree.h"

// compute the row degree
GrB_Info Degree
(
	GrB_Vector *degree,  // [output] degree vector
	GrB_Matrix A         // graph matrix
) {
	ASSERT(A      != NULL);
	ASSERT(degree != NULL);

	GrB_Info info;
	*degree = NULL;

	GrB_Vector   x              = NULL;
	GrB_Vector   out_degree     = NULL;
	GrB_Semiring plus_one_int64 = NULL;

	info = GrB_Semiring_new(&plus_one_int64, GrB_PLUS_MONOID_INT64,
			GrB_ONEB_INT64);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// determine the size of adj
	//--------------------------------------------------------------------------

	GrB_Index nrows;
	GrB_Index ncols;

	info = GrB_Matrix_nrows(&nrows, A);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_ncols(&ncols, A);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// compute the out_degree
	//--------------------------------------------------------------------------

	info = GrB_Vector_new(&out_degree, GrB_INT64, nrows);
	ASSERT(info == GrB_SUCCESS);

	// x = zeros (ncols, 1)
	info = GrB_Vector_new(&x, GrB_INT64, ncols);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_assign(x, NULL, NULL, 0, GrB_ALL, ncols, NULL);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_mxv(out_degree, NULL, NULL, plus_one_int64, A, x, NULL);
	ASSERT(info == GrB_SUCCESS);

	*degree = out_degree;

	info = GrB_free(&x);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_free(&plus_one_int64);
	ASSERT(info == GrB_SUCCESS);

	return GrB_SUCCESS;
}

