/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"
#include "../../util/rmalloc.h"

// get the fully synced GrB_Matrix from Delta_Matrix C without modifying C
GrB_Info Delta_Matrix_export
(
    GrB_Matrix *A,                  // output Matrix
    const Delta_Matrix C,           // input Delta Matrix
    const GrB_Type type,            // output matrix type (values will be typecast)
    const GrB_Orientation *format   // optional: if non-NULL, sets output orientation
                                    // before fill to avoid an in-place transpose later
) {
	ASSERT(C != NULL);
	ASSERT(A != NULL);
	ASSERT(type == GrB_BOOL || type == GrB_UINT64);

	GrB_Type  t;
	GrB_Index nrows;
	GrB_Index ncols;
	GrB_Index dp_nvals;
	GrB_Index dm_nvals;

	GrB_Matrix _A = NULL ;
	GrB_Matrix m  = DELTA_MATRIX_M (C) ;
	GrB_Matrix dp = DELTA_MATRIX_DELTA_PLUS (C) ;
	GrB_Matrix dm = DELTA_MATRIX_DELTA_MINUS (C) ;

	GrB_OK (GxB_Matrix_type  (&t, m)) ;
	ASSERT (type == GrB_BOOL || type == t) ;

	GrB_OK (GrB_Matrix_nrows (&nrows, m)) ;
	GrB_OK (GrB_Matrix_ncols (&ncols, m)) ;
	GrB_OK (GrB_Matrix_new   (&_A, type, nrows, ncols)) ;

	// set orientation on the empty matrix before filling so GraphBLAS stores
	// entries in the requested layout directly; avoids an in-place transpose
	// on an already-populated matrix, which races with OpenMP workers under
	// concurrent query load
	if (format != NULL) {
		GrB_OK (GrB_set (_A, (int32_t)*format, GrB_STORAGE_ORIENTATION_HINT)) ;
	}

	GrB_OK (GrB_Matrix_nvals (&dp_nvals, dp)) ;
	GrB_OK (GrB_Matrix_nvals (&dm_nvals, dm)) ;

	bool deletions = dm_nvals > 0 ;

	//--------------------------------------------------------------------------
	// perform copy and deletions if needed
	//--------------------------------------------------------------------------

	// in case there are items to delete use mask otherwise just copy
	GrB_Matrix     mask = deletions ? dm : NULL;
	GrB_Descriptor desc = deletions ? GrB_DESC_RSC : GrB_DESC_R;
    GrB_BinaryOp   add  = (type == GrB_BOOL) ? GrB_ONEB_BOOL : GxB_ANY_UINT64 ;

	// export in one go
	GrB_OK (GrB_Matrix_eWiseAdd_BinaryOp (_A, mask, NULL, add, m, dp, desc)) ;

	*A = _A ;
	return GrB_SUCCESS ;
}

