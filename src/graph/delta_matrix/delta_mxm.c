/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"
#include "delta_utils.h"

static inline void get_semiring_out_type
(
	GrB_Type *out_type,
	GrB_Semiring s
) {
    GrB_BinaryOp add_op ;

    // get the binary operator from the semiring monoid
    GrB_OK (GrB_Semiring_get_VOID (s, &add_op, GxB_MONOID_OPERATOR)) ;

    // get the output type (OTYPE) from the binary operator
	GrB_OK (GxB_BinaryOp_ztype (out_type, add_op)) ;

	GrB_OK (GrB_free (&add_op)) ;
}

// C = AB
// A should be fully synced on input
// C will be fully synced on output
GrB_Info Delta_mxm
(
	Delta_Matrix C,               // input/output matrix for results
	const GrB_Semiring semiring,  // defines '+' and '*' for A*B
	const Delta_Matrix A,         // first input:  matrix A (Must be synced)
	const Delta_Matrix B          // second input: matrix B
) {
	Delta_Matrix_mulCompatible (C, A, B) ;

	// multiply Delta_Matrix by Delta_Matrix
	// A * B
	// where A is fully synced!
	//
	// it is possible for either 'delta-plus' or 'delta-minus' to be empty
	// this operation performs: A * B by computing:
	// (A * (M + 'delta-plus'))<!'delta-minus'>

	// validate A doesn't contains entries in either delta-plus or delta-minus
	ASSERT(Delta_Matrix_Synced(A));

	// validate C doesn't contains entries in either delta-plus or delta-minus
	ASSERT(Delta_Matrix_Synced(C));

	GrB_Index nrows ;       // number of rows in result matrix
	GrB_Index ncols ;       // number of columns in result matrix 

	GrB_Index dp_nvals ;    // number of entries in A * 'dp'
	GrB_Index dm_nvals ;    // number of entries in A * 'dm'

	GrB_Matrix _A     = DELTA_MATRIX_M (A) ;
	GrB_Matrix _B     = DELTA_MATRIX_M (B) ;
	GrB_Matrix _C     = DELTA_MATRIX_M (C) ;
	GrB_Matrix dp     = DELTA_MATRIX_DELTA_PLUS  (B) ;
	GrB_Matrix dm     = DELTA_MATRIX_DELTA_MINUS (B) ;
	GrB_Matrix mask   = NULL;  // entities removed
	GrB_Matrix accum  = NULL;  // entities added

	GrB_Matrix_nvals   (&dp_nvals, dp) ;
	GrB_Matrix_nvals   (&dm_nvals, dm) ;
	Delta_Matrix_ncols (&ncols, C) ;
	Delta_Matrix_nrows (&nrows, C) ;

	if (dm_nvals > 0) {
		// compute A * 'delta-minus'
		GrB_OK (GrB_Matrix_new (&mask, GrB_BOOL, nrows, ncols)) ;
		GrB_OK (GrB_mxm (mask, NULL, NULL, GxB_ANY_PAIR_BOOL, _A, dm, NULL)) ;

		// update 'dm_nvals'
		GrB_OK (GrB_Matrix_nvals (&dm_nvals, mask)) ;
	}

	if (dp_nvals > 0) {
		// compute A * 'delta-plus'
		GrB_Type t ;
		get_semiring_out_type (&t, semiring) ;

		GrB_OK (GrB_Matrix_new (&accum, t, nrows, ncols)) ;
		GrB_OK (GrB_mxm (accum, NULL, NULL, semiring, _A, dp, NULL)) ;

		// update 'dp_nvals'
		GrB_OK (GrB_Matrix_nvals (&dp_nvals, accum)) ;
	}

	GrB_Descriptor desc      = NULL ;
	bool           additions = dp_nvals  >  0 ;
	bool           deletions = dm_nvals  >  0 ;

	if (deletions) {
		desc = GrB_DESC_RSC ;
	} else {
		GrB_free (&mask) ;
		mask = NULL ;
	}

	// compute (A * B)<!mask>
	GrB_OK (GrB_mxm (_C, mask, NULL, semiring, _A, _B, desc)) ;

	if (additions) {
		GrB_OK (GrB_eWiseAdd (_C, NULL, NULL, semiring, _C, accum, NULL)) ;
	}

	// clean up
	if (mask) {
		GrB_free (&mask) ;
	}

	if (accum) {
		GrB_free (&accum) ;
	}

	return GrB_SUCCESS ;
}

