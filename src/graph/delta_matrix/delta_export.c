/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"
#include "../../util/rmalloc.h"

GrB_Info Delta_Matrix_export
(
	GrB_Matrix *A,
	const Delta_Matrix C
) {
	ASSERT(C != NULL);
	ASSERT(A != NULL);

	GrB_Type   t;
	GrB_Index  nrows;
	GrB_Index  ncols;
	GrB_Index  dp_nvals;
	GrB_Index  dm_nvals;

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
	GrB_Matrix mask = deletions ? dm : NULL;
	GrB_Descriptor desc = deletions ? GrB_DESC_RSCT0 : GrB_DESC_RT0;
	GrB_OK(GrB_transpose(_A, mask, NULL, m, desc));
	
	//--------------------------------------------------------------------------
	// perform additions
	//--------------------------------------------------------------------------

	if(additions) {
		GrB_OK(GrB_Matrix_assign(
			_A, dp, NULL, dp, GrB_ALL, nrows, GrB_ALL, ncols, GrB_DESC_S));
	}

	*A = _A;
	return GrB_SUCCESS;
}

GrB_Info Delta_Matrix_export_structure
(
	GrB_Matrix *A,
	const Delta_Matrix C
) {
	int32_t iso = 0;
	Delta_Matrix_export_apply(A, C, GxB_ONE_BOOL);
	GrB_get(*A, &iso, GxB_ISO);
	ASSERT(iso == 1);
	return GrB_SUCCESS;
}

// A returns the GrB_Matrix equivalent of the given delta matrix C.
// it applies the given unary operator to the entries of C
GrB_Info Delta_Matrix_export_apply
(
	GrB_Matrix *A,
	const Delta_Matrix C,
	const GrB_UnaryOp op
) {
	ASSERT(C  != NULL);
	ASSERT(A  != NULL);
	ASSERT(op != NULL);

	GrB_Type  t;
	GrB_Index nrows;
	GrB_Index ncols;
	GrB_Index dp_nvals;
	GrB_Index dm_nvals;

	GrB_Matrix _A    = NULL;
	GrB_Matrix m     = DELTA_MATRIX_M(C);
	GrB_Matrix dp    = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix dm    = DELTA_MATRIX_DELTA_MINUS(C);
	char op_type_name[GxB_MAX_NAME_LEN];

	GrB_OK (GxB_Matrix_type  (&t, m));
	GrB_OK (GrB_Matrix_nrows (&nrows, m));
	GrB_OK (GrB_Matrix_ncols (&ncols, m));
	GrB_OK (GrB_Matrix_nvals (&dp_nvals, dp));
	GrB_OK (GrB_Matrix_nvals (&dm_nvals, dm));

	GrB_OK (GrB_UnaryOp_get_String (op, op_type_name, GrB_OUTP_TYPE_STRING));
	GrB_OK (GxB_Type_from_name(&t, op_type_name));
	GrB_OK (GrB_Matrix_new   (&_A, t, nrows, ncols));

	bool additions = dp_nvals > 0;
	bool deletions = dm_nvals > 0;

	//--------------------------------------------------------------------------
	// perform copy and deletions if needed
	//--------------------------------------------------------------------------
	
	// in case there are items to delete use mask otherwise just copy
	GrB_Matrix mask = deletions ? dm : NULL;
	GrB_Descriptor desc = deletions ? GrB_DESC_RSC: GrB_DESC_R;
	GrB_OK (GrB_Matrix_apply(_A, mask, NULL, op, m, desc));
	
	//--------------------------------------------------------------------------
	// perform additions
	//--------------------------------------------------------------------------

	if(additions) {
		GrB_OK(GrB_Matrix_apply( _A, dp, NULL, op, dp, GrB_DESC_S));
	}

	*A = _A;

	return GrB_SUCCESS;
}