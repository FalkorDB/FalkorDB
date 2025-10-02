/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"
#include "../../util/rmalloc.h"

// get matrix C without writing to internal matrix
GrB_Info Delta_Matrix_export
(
	GrB_Matrix *A,
	const Delta_Matrix C,
	const GrB_Type type
) {
	ASSERT(C != NULL);
	ASSERT(A != NULL);

	GrB_Type  t;
	GrB_Index nrows;
	GrB_Index ncols;
	GrB_Index dp_nvals;
	GrB_Index dm_nvals;

	GrB_Matrix _A = NULL;
	GrB_Matrix m  = DELTA_MATRIX_M(C);
	GrB_Matrix dp = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix dm = DELTA_MATRIX_DELTA_MINUS(C);

	GrB_OK (GxB_Matrix_type  (&t, m));
	GrB_OK (GrB_Matrix_nrows (&nrows, m));
	GrB_OK (GrB_Matrix_ncols (&ncols, m));
	GrB_OK (GrB_Matrix_new   (&_A, t, nrows, ncols));
	GrB_OK (GrB_Matrix_nvals (&dp_nvals, dp));
	GrB_OK (GrB_Matrix_nvals (&dm_nvals, dm));

	bool additions = dp_nvals > 0;
	bool deletions = dm_nvals > 0;

	//--------------------------------------------------------------------------
	// perform copy and deletions if needed
	//--------------------------------------------------------------------------

	// in case there are items to delete use mask otherwise just copy
	GrB_Matrix     mask = deletions ? dm : NULL;
	GrB_Descriptor desc = deletions ? GrB_DESC_RSC : GrB_DESC_R;

	// If type is boolean make the matrix true, otherwise copy values
	GrB_UnaryOp    op   = type == GrB_BOOL ? GxB_ONE_BOOL : GrB_IDENTITY_UINT64;

	GrB_OK (GrB_Matrix_apply(_A, mask, NULL, op, m, desc));
	
	//--------------------------------------------------------------------------
	// perform additions
	//--------------------------------------------------------------------------

	if(additions) {
		GrB_OK (GrB_Matrix_apply(_A, dp, NULL, op, dp, GrB_DESC_S));
	}

	*A = _A;
	return GrB_SUCCESS;
}
