/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"
#include "../../globals.h"

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
	ASSERT(Delta_Matrix_Synced(A));

	// validate C doesn't contains entries in either delta-plus or delta-minus
	ASSERT(Delta_Matrix_Synced(C));

	GrB_Index nrows;     // number of rows in result matrix
	GrB_Index ncols;     // number of columns in result matrix 
	GrB_Index dp_nvals;  // number of entries in A * 'dp'
	GrB_Index dm_nvals;  // number of entries in A * 'dm'

	GrB_Matrix _A      = DELTA_MATRIX_M(A);
	GrB_Matrix _B      = DELTA_MATRIX_M(B);
	GrB_Matrix _C      = DELTA_MATRIX_M(C);
	GrB_Matrix dp      = DELTA_MATRIX_DELTA_PLUS(B);
	GrB_Matrix dm      = DELTA_MATRIX_DELTA_MINUS(B);
	GrB_Matrix B_minus = NULL;  // _B - dm
	GrB_Matrix accum   = NULL; 
	GrB_Type   t       = NULL;

	GrB_OK(Delta_Matrix_type(&t, C));
	GrB_OK(Delta_Matrix_nrows(&nrows, C));
	GrB_OK(Delta_Matrix_ncols(&ncols, C));
	GrB_OK(GrB_Matrix_nvals(&dp_nvals, dp));
	GrB_OK(GrB_Matrix_nvals(&dm_nvals, dm));

	bool  additions  =  dp_nvals  >  0;
	bool  deletions  =  dm_nvals  >  0;

	if(additions) { 
		// compute A * 'delta-plus'
		GrB_OK(GrB_Matrix_new(&accum, t, nrows, ncols));

		// A could be aliased with C so this operation needs to be done before 
		// multiplying into C
		GrB_OK(GrB_mxm(accum, NULL, NULL, semiring, _A, dp, NULL));

		// update 'dp_nvals'
		GrB_OK(GrB_Matrix_nvals(&dp_nvals, accum));
		additions  =  dp_nvals  >  0;
	}

	if(deletions) { 
		Delta_Matrix_type(&t, B);
		Delta_Matrix_nrows(&nrows, B);
		Delta_Matrix_ncols(&ncols, B);
		// compute _B - dm
		GrB_OK(GrB_Matrix_new(&B_minus, t, nrows, ncols));

		GrB_OK(GrB_transpose(B_minus, dm, NULL, _B, GrB_DESC_SCT0));

		_B = B_minus;
	}

	// compute (A * B)
	GrB_OK(GrB_mxm(_C, NULL, NULL, semiring, _A, _B, NULL));

	if(additions) {
		GrB_OK(GrB_Matrix_eWiseAdd_Semiring(
			_C, NULL, NULL, semiring, _C, accum, NULL));
	}

	if(B_minus) GrB_free(&B_minus);
	if(accum) GrB_free(&accum);

	return GrB_SUCCESS;
}

// Does not look at dm. Assumes that any "zombie" value is '0'
// where x \otimes 0 = 0' and x + 0' = x. (AKA the semiring "zero")
// NOTE: this does not remove explicit zombies.
// To make the output matrix a proper delta matrix, either remove the zombies 
// or make dm contain all entries that are zombies.
// C = A * B
GrB_Info Delta_mxm_identity                    
(
    GrB_Matrix C,                 // output matrix: may contain zombie values
    const GrB_Semiring semiring,  // defines '+' and '*' for A*B
    const GrB_Semiring sem_2,     // defines '+' and '*' for matricies without zombies
    const GrB_Matrix A,           // first input:  matrix A
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

	GrB_Matrix   _B         = DELTA_MATRIX_M(B);
	GrB_Matrix   dp         = DELTA_MATRIX_DELTA_PLUS(B);
	GrB_Matrix   dm         = DELTA_MATRIX_DELTA_MINUS(B);
	GrB_Matrix   accum      = NULL; 
	GrB_Type     t          = NULL;

	int32_t iso = 0;

	GrB_OK (GrB_Matrix_get_INT32(A, &iso, GxB_ISO));
	GrB_OK (GrB_Matrix_nrows(&nrows, C));
	GrB_OK (GrB_Matrix_ncols(&ncols, C));
	GrB_OK (GrB_Matrix_nvals(&dp_nvals, dp));
	GrB_OK (GrB_Matrix_nvals(&dm_nvals, dm));
	GrB_OK (GxB_Matrix_type (&t, C));

	// currently only bool is supported for the output
	ASSERT(t == GrB_BOOL);
	
	bool  zombies    =  iso == 0;
	bool  additions  =  dp_nvals  >  0;
	bool  deletions  =  dm_nvals  >  0;

	if(zombies){
		bool v = true;
		GrB_OK (GrB_Scalar_extractElement_BOOL(&v, (GrB_Scalar) C));
		zombies = !v;
	}

	GrB_Semiring DP_semiring = zombies ? semiring : sem_2;
	GrB_Semiring M_semiring  = deletions || zombies ? semiring : sem_2;

	if(additions) { 
		// compute A * 'delta-plus'
		GrB_OK(GrB_Matrix_new(&accum, t, nrows, ncols));

		// A could be aliased with C, so this operation needs to be done before 
		// multiplying into C
		GrB_OK(GrB_mxm(accum, NULL, NULL, DP_semiring, A, dp, NULL));

		// update 'dp_nvals'
		GrB_OK(GrB_Matrix_nvals(&dp_nvals, accum));
		additions  =  dp_nvals  >  0;
	}

	// compute (A * B)
	GrB_OK(GrB_mxm(C, NULL, NULL, M_semiring, A, _B, NULL));

	if(additions) {
		GrB_OK(GrB_Matrix_eWiseAdd_Semiring(
			C, NULL, NULL, DP_semiring, C, accum, NULL));
	}

	GrB_free(&accum);
	return GrB_SUCCESS;
}

#if 0
GrB_Info Delta_mxm_struct
(
    GrB_Matrix C,                 // output: matrix C 
    const GrB_Matrix A,           // first input:  matrix A
    const Delta_Matrix B          // second input: matrix B
) {
	ASSERT(C != NULL);
	ASSERT(A != NULL);
	ASSERT(B != NULL);


	GrB_Index nrows;     // number of rows in result matrix
	GrB_Index ncols;     // number of columns in result matrix 
	GrB_Index dp_nvals;
	GrB_Index dm_nvals;
	bool 	  additions;
	bool      deletions;

	GrB_Matrix    m        = DELTA_MATRIX_M(B);
	GrB_Matrix    dp       = DELTA_MATRIX_DELTA_PLUS(B);
	GrB_Matrix    dm       = DELTA_MATRIX_DELTA_MINUS(B);
	GrB_Matrix    _C       = NULL; 
	GxB_Container cont     = NULL;
	GrB_Scalar    temp     = NULL;

	GrB_OK(GrB_Matrix_nrows(&nrows, C));
	GrB_OK(GrB_Matrix_ncols(&ncols, C));
	GrB_OK(GrB_Matrix_nvals(&dp_nvals, dp));
	GrB_OK(GrB_Matrix_nvals(&dm_nvals, dm));
	additions  =  dp_nvals  >  0;
	deletions  =  dm_nvals  >  0;

	if(deletions) {
		GrB_Semiring  semiring = GxB_PLUS_PAIR_UINT64;
		GrB_OK(GrB_Matrix_new(&_C, GrB_UINT64, nrows, ncols));
		GrB_OK(GxB_Container_new(&cont));

		// compute (A * BM)
		GrB_OK(GrB_mxm(_C, NULL, NULL, semiring, A, m, NULL));

		// C -= (A * BDM) 
		GrB_OK(GrB_mxm(_C, NULL, GrB_MINUS_UINT64, semiring, A, dm, NULL));

		// C += (A * BDP)
		GrB_OK(GrB_mxm(_C, NULL, GrB_PLUS_UINT64, semiring, A, dp, NULL));

		GrB_OK(GrB_Matrix_select_UINT64(
			_C, NULL, NULL, GrB_VALUENE_UINT64, _C, (uint64_t) 0, NULL));
		GrB_OK(GrB_Matrix_assign_BOOL(
			_C, _C, NULL, true, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S));

		// move _C into C
		GrB_OK(GxB_unload_Matrix_into_Container(_C, cont, NULL));
		GrB_OK(GxB_load_Matrix_from_Container(C, cont, NULL));
	} else {
		GrB_Semiring  semiring = GxB_ANY_PAIR_BOOL;
		GrB_OK(GrB_Matrix_new(&_C, GrB_BOOL, nrows, ncols));

		if(additions){
			// _C = (A * BDP)
			GrB_OK(GrB_mxm(_C, NULL, NULL, semiring, A, dp, NULL));
		}

		// compute (A * BM)
		GrB_OK(GrB_mxm(C, NULL, NULL, semiring, A, m, NULL));

		if(additions){
			// C += _C
			GrB_OK (GrB_Matrix_assign_BOOL(
				C, _C, NULL, true, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S));
		}
	}
	GrB_free(&_C);
	return GrB_SUCCESS;
}
#else
GrB_Info Delta_mxm_struct
(
    GrB_Matrix C,                 // output: matrix C 
    const GrB_Matrix A,           // first input:  matrix A
    const Delta_Matrix B          // second input: matrix B
) {
	ASSERT(C != NULL);
	ASSERT(A != NULL);
	ASSERT(B != NULL);


	GrB_Index nrows;     // number of rows in result matrix
	GrB_Index ncols;     // number of columns in result matrix 
	GrB_Index dp_nvals;
	GrB_Index dm_nvals;
	bool 	  additions;
	bool      deletions;

	GrB_Matrix    m        = DELTA_MATRIX_M(B);
	GrB_Matrix    dp       = DELTA_MATRIX_DELTA_PLUS(B);
	GrB_Matrix    dm       = DELTA_MATRIX_DELTA_MINUS(B);
	GrB_Matrix    mask     = NULL; 
	GrB_Matrix    _C       = NULL; 
	GxB_Container cont     = NULL;
	GrB_Scalar    temp     = NULL;

	GrB_OK(GrB_Matrix_nrows(&nrows, C));
	GrB_OK(GrB_Matrix_ncols(&ncols, C));
	GrB_OK(GrB_Matrix_nvals(&dp_nvals, dp));
	GrB_OK(GrB_Matrix_nvals(&dm_nvals, dm));
	additions  =  dp_nvals  >  0;
	deletions  =  dm_nvals  >  0;

	if(deletions) {
		GrB_OK(GrB_Matrix_new(&mask, GrB_UINT64, nrows, ncols));

		// C -= (A * BDM) 
		GrB_OK(GrB_mxm(mask, NULL, NULL, GxB_PLUS_PAIR_UINT64, A, dm, NULL));

		// check if there are any deletions
		GrB_OK (GrB_Matrix_nvals(&dm_nvals, mask));
		deletions  =  dm_nvals  >  0;
		if (deletions){
			GrB_OK(GrB_Matrix_apply(
				mask, NULL, NULL, GrB_AINV_UINT64, mask, NULL));
			// compute (A * BM)
			GrB_OK(GrB_mxm(mask, mask, GrB_PLUS_UINT64, GxB_PLUS_PAIR_UINT64, 
				A, m, GrB_DESC_S));
			GrB_OK (GrB_Matrix_select_UINT64(mask, NULL, NULL,
				GrB_VALUEEQ_UINT64, mask, (uint64_t) 0, NULL));
			
			// check if there are any deletions
			GrB_OK (GrB_Matrix_nvals(&dm_nvals, mask));
			deletions  =  dm_nvals  >  0;
		}
		if (!deletions){
			GrB_free(&mask);
		}
	}

	GrB_Semiring  semiring = GxB_ANY_PAIR_BOOL;
	GrB_OK(GrB_Matrix_new(&_C, GrB_BOOL, nrows, ncols));

	if(additions){

		// _C = (A * BDP)
		GrB_OK(GrB_mxm(_C, NULL, NULL, semiring, A, dp, NULL));
	}

	// compute (A * BM)
	// GrB_Descriptor desc = deletions ? GrB_DESC_RSC : NULL;
	// GrB_OK(GrB_mxm(C, mask, NULL, semiring, A, m, desc));

	GrB_OK(GrB_mxm(C, NULL, NULL, semiring, A, m, NULL));

	if(deletions){
		GrB_OK (GrB_Matrix_assign_Scalar(C, mask, NULL, Global_GrB_Ops_Get()->empty,
			GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S));
	}

	if(additions){
		// C += _C
		GrB_OK (GrB_Matrix_assign_BOOL(
			C, _C, NULL, true, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S));
	}

	GrB_free(&_C);
	GrB_free(&mask);
	return GrB_SUCCESS;
}
#endif

#if 1
// Using a plus_x semiring, returns C = A (BM + BMP - BDM)
// Note this method can be tweeked to be used for any monoid with an inverse 
// operation
// TODO: make a better name for this function. 
GrB_Info Delta_mxm_count
(
    GrB_Matrix C,                 // output: matrix C 
    const GrB_Semiring semiring,  // defines '+' and '*' for A*B
    const GrB_Matrix A,           // first input:  matrix A
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

	GrB_Index nrows;     // number of rows in result matrix
	GrB_Index ncols;     // number of columns in result matrix 
	GrB_Index dp_nvals;  // number of entries in A * 'dp'
	GrB_Index dm_nvals;  // number of entries in A * 'dm'

	GrB_Matrix    _B     = DELTA_MATRIX_M(B);
	GrB_Matrix    dp     = DELTA_MATRIX_DELTA_PLUS(B);
	GrB_Matrix    dm     = DELTA_MATRIX_DELTA_MINUS(B);
	GrB_Matrix    _C     = NULL; 
	GxB_Container cont   = NULL;
	GrB_Vector    temp   = NULL;

	GrB_OK(GrB_Matrix_nrows(&nrows, C));
	GrB_OK(GrB_Matrix_ncols(&ncols, C));
	GrB_OK(GrB_Matrix_nvals(&dp_nvals, dp));
	GrB_OK(GrB_Matrix_new(&_C, GrB_UINT64, nrows, ncols));

	// compute (A * BM)
	GrB_OK(GrB_mxm(_C, NULL, NULL, semiring, A, _B, NULL));

	// C -= (A * BDM) 
	GrB_OK(GrB_mxm(_C, NULL, GrB_MINUS_UINT64, semiring, A, dm, NULL));

	// C += (A * BDP)
	GrB_OK(GrB_mxm(_C, NULL, GrB_PLUS_UINT64, semiring, A, dp, NULL));

	GrB_OK(GrB_Matrix_assign_BOOL(
		C, _C, NULL, true, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_R));

	GrB_free(&_C);

	return GrB_SUCCESS;
}
#else // C and A uint 64
// Using a plus_x semiring, returns C = A (BM + BMP - BDM)
// Note this method can be tweeked to be used for any monoid with an inverse 
// operation
GrB_Info Delta_mxm_count
(
    GrB_Matrix C,                 // output: matrix C 
    const GrB_Semiring semiring,  // defines '+' and '*' for A*B
    const GrB_Matrix A,           // first input:  matrix A
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
	GrB_Matrix  _C       =  NULL; 
	GrB_Type    t        =  NULL;
	bool        aliased  =  C == A;

	GrB_Matrix_nrows(&nrows, C);
	GrB_Matrix_ncols(&ncols, C);
	GrB_Matrix_nvals(&dp_nvals, dp);
	GxB_Matrix_type (&t, C);

	if(aliased) {
		info = GrB_Matrix_new(&_C, t, nrows, ncols);
		ASSERT(info == GrB_SUCCESS);
		C = _C;
	}

	// compute (A * BM)
	info = GrB_mxm(C, NULL, NULL, semiring, A, _B, NULL);
	ASSERT(info == GrB_SUCCESS);

	// C -= (A * BDM) (The mask is used to help GBLAS, does not affect outcome).
	info = GrB_mxm(C, C, GrB_MINUS_UINT64, semiring, A, dm, GrB_DESC_S);
	ASSERT(info == GrB_SUCCESS);

	// C += (A * BDP)
	info = GrB_mxm(C, NULL, GrB_PLUS_UINT64, semiring, A, dp, NULL);
	ASSERT(info == GrB_SUCCESS);

	
	// uncomment to remove zeros
	GrB_Matrix_select_UINT64(C, NULL, NULL, GrB_VALUEGT_BOOL, C, 0, NULL);

	if(aliased) {
		info = GrB_transpose(A, NULL, NULL, _C, GrB_DESC_T0);
		ASSERT(info == GrB_SUCCESS);
		GrB_free(&_C);
	}

	return info;
}
#endif
