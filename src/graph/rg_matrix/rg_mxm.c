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

GrB_Info RG_mxm                     // C = A * B
(
    RG_Matrix C,                    // input/output matrix for results
    const GrB_Semiring semiring,    // defines '+' and '*' for A*B
    const RG_Matrix A,              // first input:  matrix A
    const RG_Matrix B               // second input: matrix B
) {
	ASSERT(C != NULL);
	ASSERT(A != NULL);
	ASSERT(B != NULL);

	// multiply RG_Matrix by RG_Matrix
	// A * B
	// where A is fully synced!
	//
	// it is possible for either 'delta-plus' or 'delta-minus' to be empty
	// this operation performs: A * B by computing:
	// (A * (M + 'delta-plus'))<!'delta-minus'>

	// validate A doesn't contains entries in either delta-plus or delta-minus
	ASSERT(RG_Matrix_Synced(A));

	// validate C doesn't contains entries in either delta-plus or delta-minus
	ASSERT(RG_Matrix_Synced(C));

	GrB_Info info;
	GrB_Index nrows;     // number of rows in result matrix
	GrB_Index ncols;     // number of columns in result matrix 
	GrB_Index dp_nvals;  // number of entries in A * 'dp'
	GrB_Index dm_nvals;  // number of entries in A * 'dm'

	GrB_Matrix  _A     =  RG_MATRIX_M(A);
	GrB_Matrix  _B     =  RG_MATRIX_M(B);
	GrB_Matrix  _C     =  RG_MATRIX_M(C);
	GrB_Matrix  dp     =  RG_MATRIX_DELTA_PLUS(B);
	GrB_Matrix  dm     =  RG_MATRIX_DELTA_MINUS(B);
	GrB_Matrix  mask   =  NULL;  // entities removed
	GrB_Matrix  accum  =  NULL;  // entities added

	RG_Matrix_nrows(&nrows, C);
	RG_Matrix_ncols(&ncols, C);
	GrB_Matrix_nvals(&dp_nvals, dp);
	GrB_Matrix_nvals(&dm_nvals, dm);

	if(dm_nvals > 0) {
		// compute A * 'delta-minus'
		info = GrB_Matrix_new(&mask, GrB_BOOL, nrows, ncols);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_mxm(mask, NULL, NULL, GxB_ANY_PAIR_BOOL, _A, dm, NULL);
		ASSERT(info == GrB_SUCCESS);

		// update 'dm_nvals'
		info = GrB_Matrix_nvals(&dm_nvals, mask);
		ASSERT(info == GrB_SUCCESS);
	}

	if(dp_nvals > 0) {
		// compute A * 'delta-plus'
		info = GrB_Matrix_new(&accum, GrB_BOOL, nrows, ncols);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_mxm(accum, NULL, NULL, semiring, _A, dp, NULL);
		ASSERT(info == GrB_SUCCESS);

		// update 'dp_nvals'
		info = GrB_Matrix_nvals(&dp_nvals, accum);
		ASSERT(info == GrB_SUCCESS);
	}

	GrB_Descriptor  desc       =  NULL;
	bool            additions  =  dp_nvals  >  0;
	bool            deletions  =  dm_nvals  >  0;

	if (deletions) {
		desc = GrB_DESC_RSC;
	} else {
		GrB_free(&mask);
		mask = NULL;
	}

	// compute (A * B)<!mask>
	info = GrB_mxm(_C, mask, NULL, semiring, _A, _B, desc);
	ASSERT(info == GrB_SUCCESS);

	if(additions) {
		info = GrB_eWiseAdd(_C, NULL, NULL, GxB_ANY_PAIR_BOOL, _C, accum, NULL);
		ASSERT(info == GrB_SUCCESS);
	}

	// clean up
	if(mask)  GrB_free(&mask);
	if(accum) GrB_free(&accum);

	return info;
}

