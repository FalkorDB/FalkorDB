/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"

// C = A * B
GrB_Info Delta_mxm                     
(
    Delta_Matrix C,               // input/output matrix for results
    const GrB_Semiring semiring,  // defines '+' and '*' for A*B
    const Delta_Matrix A,         // first input:  matrix A
    const Delta_Matrix B          // second input: matrix B
) {
	ASSERT(C != NULL);
	ASSERT(A != NULL);
	ASSERT(B != NULL);
	ASSERT(C != B);

	// multiply Delta_Matrix by Delta_Matrix
	// A * B
	// where A is fully synced!
	//
	// it is possible for either 'delta-plus' or 'delta-minus' to be empty
	// this operation performs: A * B by computing:
	// (A * (M + 'delta-plus'))<!'delta-minus'>

	// validate A doesn't contains entries in either delta-plus or delta-minus
	assert(Delta_Matrix_Synced(A));

	// validate C doesn't contains entries in either delta-plus or delta-minus
	assert(Delta_Matrix_Synced(C));

	GrB_Info info;
	GrB_Index nrows;     // number of rows in result matrix
	GrB_Index ncols;     // number of columns in result matrix 
	GrB_Index dp_nvals;  // number of entries in A * 'dp'
	GrB_Index dm_nvals;  // number of entries in A * 'dm'

	GrB_Matrix  _A       =  DELTA_MATRIX_M(A);
	GrB_Matrix  _B       =  DELTA_MATRIX_M(B);
	GrB_Matrix  _C       =  DELTA_MATRIX_M(C);
	GrB_Matrix  dp       =  DELTA_MATRIX_DELTA_PLUS(B);
	GrB_Matrix  dm       =  DELTA_MATRIX_DELTA_MINUS(B);
	GrB_Matrix  B_minus  =  NULL;  // _B - dm
	GrB_Matrix  accum    =  NULL; 
	GrB_Type    t        =  NULL;

	Delta_Matrix_type(&t, C);
	Delta_Matrix_nrows(&nrows, C);
	Delta_Matrix_ncols(&ncols, C);
	GrB_Matrix_nvals(&dp_nvals, dp);
	GrB_Matrix_nvals(&dm_nvals, dm);

	bool  additions  =  dp_nvals  >  0;
	bool  deletions  =  dm_nvals  >  0;
	// printf("USING MXM\n");

	if(additions) { 
		// compute A * 'delta-plus'
		// printf("Handle dp\n");
		info = GrB_Matrix_new(&accum, t, nrows, ncols);
		ASSERT(info == GrB_SUCCESS);

		// A could be aliased with C so this operation needs to be done before 
		// multiplying into C
		info = GrB_mxm(accum, NULL, NULL, semiring, _A, dp, NULL);
		ASSERT(info == GrB_SUCCESS);

		// update 'dp_nvals'
		info = GrB_Matrix_nvals(&dp_nvals, accum);
		ASSERT(info == GrB_SUCCESS);
		additions  =  dp_nvals  >  0;
	}

	if(deletions) { 
		// printf("Handle dm\n");
		Delta_Matrix_type(&t, B);
		Delta_Matrix_nrows(&nrows, B);
		Delta_Matrix_ncols(&ncols, B);
		// compute _B - dm
		info = GrB_Matrix_new(&B_minus, t, nrows, ncols);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_transpose(B_minus, dm, NULL, _B, GrB_DESC_SCT0);
		ASSERT(info == GrB_SUCCESS);

		_B = B_minus;
	}

	// compute (A * B)
	info = GrB_mxm(_C, NULL, NULL, semiring, _A, _B, NULL);
	ASSERT(info == GrB_SUCCESS);

	if(additions) {
		info = GrB_Matrix_eWiseAdd_Semiring(
			_C, NULL, NULL, semiring, _C, accum, NULL);
		ASSERT(info == GrB_SUCCESS);
	}

	if(B_minus) GrB_free(&B_minus);
	if(accum) GrB_free(&accum);

	return info;
}

// Does not look at dm. Assumes that any "zombie" value is '0'
// where x \otimes 0 = 0' and x + 0' = x. (AKA the semiring "zero")
// NOTE: this does not remove explicit zombies.
// To make the output matrix a proper delta matrix, either remove the zombies 
// or make dm contain all entries that are zombies.
// C = A * B
GrB_Info Delta_mxm_identity                    
(
    GrB_Matrix C,                 // input/output matrix for results may contain zombie values
    const GrB_Semiring semiring,  // defines '+' and '*' for A*B
    const GrB_Matrix A,           // first input:  matrix A may contain zombie values
    const Delta_Matrix B          // second input: matrix B
) {
	ASSERT(C != NULL);
	ASSERT(A != NULL);
	ASSERT(B != NULL);

	// multiply Delta_Matrix by Delta_Matrix
	// A * B
	// where A is fully synced!
	//
	// it is possible for either 'delta-plus' or 'delta-minus' to be empty
	// this operation performs: A * B by computing:
	// (A * (M + 'delta-plus')) 
	// it requires that zombie values are the zero of the inputed semiring
	// it will output a GrB_Matrix that may contain zombie values

	GrB_Info info;
	GrB_Index nrows;     // number of rows in result matrix
	GrB_Index ncols;     // number of columns in result matrix 
	GrB_Index dp_nvals;  // number of entries in A * 'dp'
	GrB_Index dm_nvals;  // number of entries in A * 'dm'

	GrB_Matrix  _B       =  DELTA_MATRIX_M(B);
	GrB_Matrix  dp       =  DELTA_MATRIX_DELTA_PLUS(B);
	GrB_Matrix  dm       =  DELTA_MATRIX_DELTA_MINUS(B);
	GrB_Matrix  B_minus  =  NULL;  // _B - dm
	GrB_Matrix  accum    =  NULL; 
	GrB_Type    t        =  NULL;

	GrB_Matrix_nrows(&nrows, C);
	GrB_Matrix_ncols(&ncols, C);
	GrB_Matrix_nvals(&dp_nvals, dp);
	GxB_Matrix_type (&t, C);
	
	bool  additions  =  dp_nvals  >  0;

	if(additions) { 
		// compute A * 'delta-plus'
		info = GrB_Matrix_new(&accum, t, nrows, ncols);
		ASSERT(info == GrB_SUCCESS);

		// A could be aliased with C, so this operation needs to be done before 
		// multiplying into C
		info = GrB_mxm(accum, NULL, NULL, semiring, A, dp, NULL);
		ASSERT(info == GrB_SUCCESS);

		// update 'dp_nvals'
		info = GrB_Matrix_nvals(&dp_nvals, accum);
		ASSERT(info == GrB_SUCCESS);
		additions  =  dp_nvals  >  0;
	}

	// compute (A * B)
	info = GrB_mxm(C, NULL, NULL, semiring, A, _B, NULL);
	ASSERT(info == GrB_SUCCESS);

	if(additions) {
		info = GrB_Matrix_eWiseAdd_Semiring(
			C, NULL, NULL, semiring, C, accum, NULL);
		ASSERT(info == GrB_SUCCESS);
	}
	
	// uncomment to remove zombies
	// GrB_Matrix_select_BOOL(C, NULL, NULL, GrB_VALUEEQ_BOOL, C, true, NULL);

	if(B_minus) GrB_free(&B_minus);
	if(accum) GrB_free(&accum);

	return info;
}
