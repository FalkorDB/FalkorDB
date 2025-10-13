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

GrB_Info Delta_mxm_struct
(
    Delta_Matrix C,               // output: matrix C 
    const Delta_Matrix A,         // first input:  matrix A
    const Delta_Matrix B          // second input: matrix B
) {
	ASSERT(C != NULL);
	ASSERT(A != NULL);
	ASSERT(B != NULL);

	// multiply Delta_Matrix by Delta_Matrix
	// A * B
	// where A has no DP entries

	GrB_Index nrows;     // number of rows in result matrix
	GrB_Index ncols;     // number of columns in result matrix 
	GrB_Index dp_nvals;  // number of entries in A * 'dp'
	GrB_Index dm_nvals;  // number of entries in A * 'dm'

	GrB_Matrix CM         = DELTA_MATRIX_M(C);
	GrB_Matrix CDP        = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix CDM        = DELTA_MATRIX_DELTA_MINUS(C);
	GrB_Matrix AM         = DELTA_MATRIX_M(A);
	GrB_Matrix ADP        = DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix ADM        = DELTA_MATRIX_DELTA_MINUS(A);
	GrB_Matrix BM         = DELTA_MATRIX_M(B);
	GrB_Matrix BDP        = DELTA_MATRIX_DELTA_PLUS(B);
	GrB_Matrix BDM        = DELTA_MATRIX_DELTA_MINUS(B);
	GrB_Matrix maybe_del  = NULL;
	GrB_Matrix accum      = NULL;

	GrB_OK (GrB_Matrix_nvals(&dp_nvals, ADP));
	ASSERT(dp_nvals == 0);
	
	GrB_OK (GrB_Matrix_nvals(&dm_nvals, ADM));
	bool deletions_a = dm_nvals > 0;

	GrB_OK (GrB_Matrix_nvals(&dp_nvals, BDP));
	bool additions_b = dp_nvals > 0;

	GrB_OK (GrB_Matrix_nvals(&dm_nvals, BDM));
	bool deletions_b = dm_nvals > 0;

	GrB_OK(GrB_Matrix_nrows(&nrows, CM));
	GrB_OK(GrB_Matrix_ncols(&ncols, CM));

	if(additions_b) {
		GrB_OK (GrB_Matrix_new(&accum, GrB_BOOL, nrows, ncols));
		GrB_OK (GrB_mxm(accum, NULL, NULL, GxB_ANY_PAIR_BOOL, AM, BDP, NULL));
		GrB_OK (GrB_Matrix_nvals(&dp_nvals, accum));
		if(dp_nvals == 0) {
			GrB_free(&accum);
			additions_b = false;
		}
	}

	// compute (AM * BM)
	GrB_OK(GrB_mxm(CM, NULL, NULL, GxB_ANY_PAIR_BOOL, AM, BM, NULL));

	if(accum != NULL) {
		GrB_OK(GrB_Matrix_eWiseAdd_BinaryOp(
			CM, NULL, NULL, GrB_ONEB_BOOL, CM, accum, NULL));
	}

	if(deletions_a || deletions_b){
		GrB_OK(GrB_Matrix_new(&maybe_del, GrB_UINT64, nrows, ncols));

		if(deletions_b) {
			GrB_OK(GrB_mxm(
				maybe_del, NULL, NULL, GxB_PLUS_PAIR_UINT64, AM, BDM, NULL));
		}

		if(deletions_a) {
			GrB_OK(GrB_mxm(maybe_del, NULL, GrB_PLUS_UINT64, 
				GxB_PLUS_PAIR_UINT64, ADM, BM, NULL));
			GrB_OK(GrB_mxm(maybe_del, NULL, GrB_PLUS_UINT64, 
				GxB_PLUS_PAIR_UINT64, ADM, BDP, NULL));
		}

		if(deletions_b && deletions_a) {
			GrB_OK(GrB_mxm(maybe_del, maybe_del, GrB_MINUS_UINT64, 
				GxB_PLUS_PAIR_UINT64, ADM, BDM, GrB_DESC_S));
		}

		GrB_OK(GrB_mxm(maybe_del, maybe_del, GrB_MINUS_UINT64, 
			GxB_PLUS_PAIR_UINT64, AM, BM, GrB_DESC_S));

		if(accum) {
			GrB_OK(GrB_mxm(maybe_del, maybe_del, GrB_MINUS_UINT64, 
				GxB_PLUS_PAIR_UINT64, AM, BDP, GrB_DESC_S));
		}
		GrB_OK(GrB_Matrix_select_UINT64(maybe_del, NULL, NULL, GrB_VALUEEQ_INT64, 
			maybe_del, 0, NULL));
		GrB_OK(GrB_Matrix_assign_BOOL(
			CDM, maybe_del, NULL, true, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_RS));
	}

	GrB_free(&accum);
	GrB_free(&maybe_del);

	return GrB_SUCCESS;
}

GrB_Info Delta_mxm_struct_V2
(
    Delta_Matrix C,               // output: matrix C 
    const GrB_Semiring semiring,  // defines '+' and '*' for A*B
    const Delta_Matrix A,         // first input:  matrix A
    const Delta_Matrix B          // second input: matrix B
) {
	ASSERT(C != NULL);
	ASSERT(A != NULL);
	ASSERT(B != NULL);

	// multiply Delta_Matrix by Delta_Matrix
	// A * B
	// where A has no DP entries

	GrB_Index nrows;     // number of rows in result matrix
	GrB_Index ncols;     // number of columns in result matrix 
	GrB_Index dp_nvals;  // number of entries in A * 'dp'
	GrB_Index dm_nvals;  // number of entries in A * 'dm'

	GrB_Matrix CM         = DELTA_MATRIX_M(C);
	GrB_Matrix CDP        = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix CDM        = DELTA_MATRIX_DELTA_MINUS(C);
	GrB_Matrix AM         = DELTA_MATRIX_M(A);
	GrB_Matrix ADP        = DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix ADM        = DELTA_MATRIX_DELTA_MINUS(A);
	GrB_Matrix BM         = DELTA_MATRIX_M(B);
	GrB_Matrix BDP        = DELTA_MATRIX_DELTA_PLUS(B);
	GrB_Matrix BDM        = DELTA_MATRIX_DELTA_MINUS(B);
	GrB_Matrix maybe_del  = NULL;
	GrB_Matrix accum      = NULL;

	GrB_OK (GrB_Matrix_nvals(&dp_nvals, ADP));
	ASSERT(dp_nvals == 0);
	
	GrB_OK (GrB_Matrix_nvals(&dm_nvals, ADM));
	bool deletions_a = dm_nvals > 0;

	GrB_OK (GrB_Matrix_nvals(&dp_nvals, BDP));
	bool additions_b = dp_nvals > 0;

	GrB_OK (GrB_Matrix_nvals(&dm_nvals, BDM));
	bool deletions_b = dm_nvals > 0;

	GrB_OK(GrB_Matrix_nrows(&nrows, CM));
	GrB_OK(GrB_Matrix_ncols(&ncols, CM));

	if(additions_b) {
		GrB_OK (GrB_Matrix_new(&accum, GrB_BOOL, nrows, ncols));
		GrB_OK (GrB_mxm(accum, NULL, NULL, semiring, AM, BDP, NULL));
		GrB_OK (GrB_Matrix_nvals(&dp_nvals, accum));
		if(dp_nvals == 0) {
			GrB_free(&accum);
			additions_b = false;
		}
	}

	// compute (AM * BM)
	GrB_OK(GrB_mxm(CM, NULL, NULL, semiring, AM, BM, NULL));

	if(accum != NULL) {
		GrB_OK(GrB_Matrix_eWiseAdd_BinaryOp(
			CM, NULL, NULL, GrB_LOR, CM, accum, NULL));
	}

	if(deletions_a || deletions_b){
		// GrB_OK(GrB_Matrix_new(&maybe_del, GrB_BOOL, nrows, ncols));

		// if(deletions_b) {
		// 	GrB_OK(GrB_mxm(
		// 		maybe_del, NULL, NULL, GxB_ANY_PAIR_BOOL, AM, BDM, NULL));
		// }

		// if(deletions_a) {
		// 	GrB_OK(GrB_mxm(maybe_del, NULL, GrB_ONEB_BOOL, 
		// 		GxB_ANY_PAIR_BOOL, ADM, BM, NULL));
		// 	GrB_OK(GrB_mxm(maybe_del, NULL, GrB_ONEB_BOOL, 
		// 		GxB_ANY_PAIR_BOOL, ADM, BDP, NULL));
		// }

		// GrB_OK(GrB_Matrix_assign(maybe_del, maybe_del, NULL, CM, GrB_ALL, 0, GrB_ALL, 0, NULL));
		GrB_OK (GrB_Matrix_select_BOOL(CM, NULL, NULL, GrB_VALUENE_BOOL, 
			CM, BOOL_ZOMBIE, NULL));

		// GrB_OK(GrB_Matrix_select_BOOL(CDM, NULL, NULL, GrB_VALUEEQ_BOOL, 
		// 	CM, BOOL_ZOMBIE, NULL));

		// GrB_OK(GrB_Matrix_assign_BOOL(
		// 	CDM, CDM, NULL, true, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_RS));
	}

	GrB_free(&accum);
	GrB_free(&maybe_del);

	return GrB_SUCCESS;
}

GrB_Info Delta_mxm_struct_V3
(
    Delta_Matrix C,               // output: matrix C 
    const GrB_Semiring semiring,  // defines '+' and '*' for A*B
    const Delta_Matrix A,         // first input:  matrix A
    const Delta_Matrix B          // second input: matrix B
) {
	ASSERT(C != NULL);
	ASSERT(A != NULL);
	ASSERT(B != NULL);

	// multiply Delta_Matrix by Delta_Matrix
	// A * B
	// where A has no DP entries

	GrB_Index nrows;     // number of rows in result matrix
	GrB_Index ncols;     // number of columns in result matrix 
	GrB_Index dp_nvals;  // number of entries in A * 'dp'
	GrB_Index dm_nvals;  // number of entries in A * 'dm'

	GrB_Matrix CM         = DELTA_MATRIX_M(C);
	GrB_Matrix CDP        = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix CDM        = DELTA_MATRIX_DELTA_MINUS(C);
	GrB_Matrix AM         = DELTA_MATRIX_M(A);
	GrB_Matrix ADP        = DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix ADM        = DELTA_MATRIX_DELTA_MINUS(A);
	GrB_Matrix BM         = DELTA_MATRIX_M(B);
	GrB_Matrix BDP        = DELTA_MATRIX_DELTA_PLUS(B);
	GrB_Matrix BDM        = DELTA_MATRIX_DELTA_MINUS(B);
	GrB_Matrix maybe_del  = NULL;
	GrB_Matrix accum      = NULL;
	GrB_BinaryOp op       = NULL;
	GrB_Semiring sem      = NULL;

	GrB_Semiring_get_VOID(semiring, (void *)&op, GxB_MONOID_OPERATOR);

	GrB_OK (GrB_Matrix_nvals(&dp_nvals, ADP));
	bool additions_a = dp_nvals > 0;
	
	GrB_OK (GrB_Matrix_nvals(&dm_nvals, ADM));
	bool deletions_a = dm_nvals > 0;

	GrB_OK (GrB_Matrix_nvals(&dp_nvals, BDP));
	bool additions_b = dp_nvals > 0;

	GrB_OK (GrB_Matrix_nvals(&dm_nvals, BDM));
	bool deletions_b = dm_nvals > 0;

	GrB_OK(GrB_Matrix_nrows(&nrows, CM));
	GrB_OK(GrB_Matrix_ncols(&ncols, CM));

	if(additions_a || additions_b) {
		GrB_OK (GrB_Matrix_new(&accum, GrB_BOOL, nrows, ncols));
		GrB_set(accum, GxB_HYPERSPARSE, GxB_SPARSITY_CONTROL);
	}

	if(additions_a) {
		sem = deletions_b ? semiring : GxB_ANY_PAIR_BOOL;
		GrB_OK (GrB_mxm(accum, NULL, NULL, sem, ADP, BM, NULL));
	}
	
	if(additions_b) {
		sem = deletions_a ? semiring : GxB_ANY_PAIR_BOOL;
		GrB_OK (GrB_mxm(accum, NULL, op, sem, AM, BDP, NULL));
	}

	if(additions_a && additions_b) {
		GrB_OK (GrB_mxm(accum, NULL, op, semiring, ADP, BDP, NULL));
	}

	sem = (deletions_a || deletions_b) ? semiring : GxB_ANY_PAIR_BOOL;
	// compute (AM * BM)
	GrB_OK(GrB_mxm(CM, NULL, NULL, sem, AM, BM, NULL));

	if(accum != NULL) {
		GrB_OK(GrB_Matrix_select_BOOL(CDP, NULL, NULL, GrB_VALUENE_BOOL, 
			accum, BOOL_ZOMBIE, NULL));
		
		GrB_OK (GrB_Matrix_eWiseMult_BinaryOp(
			accum, NULL, NULL, op, CM, accum, NULL));

		GrB_OK (GrB_Matrix_assign_Scalar(
			CDP, accum, NULL, Global_GrB_Ops_Get()->empty, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S));

		GrB_OK (GrB_Matrix_assign_BOOL(
			CM, accum, NULL, true, GrB_ALL, 0, GrB_ALL, 0, NULL));
	}

	if(deletions_a || deletions_b){
	// 	// GrB_OK(GrB_Matrix_new(&maybe_del, GrB_BOOL, nrows, ncols));

	// 	// if(deletions_b) {
	// 	// 	GrB_OK(GrB_mxm(
	// 	// 		maybe_del, NULL, NULL, GxB_ANY_PAIR_BOOL, AM, BDM, NULL));
	// 	// }

	// 	// if(deletions_a) {
	// 	// 	GrB_OK(GrB_mxm(maybe_del, NULL, GrB_ONEB_BOOL, 
	// 	// 		GxB_ANY_PAIR_BOOL, ADM, BM, NULL));
	// 	// 	GrB_OK(GrB_mxm(maybe_del, NULL, GrB_ONEB_BOOL, 
	// 	// 		GxB_ANY_PAIR_BOOL, ADM, BDP, NULL));
	// 	// }

	// 	// GrB_OK(GrB_Matrix_assign(maybe_del, maybe_del, NULL, CM, GrB_ALL, 0, GrB_ALL, 0, NULL));
	// 	// GrB_OK (GrB_Matrix_select_BOOL(CM, NULL, NULL, GrB_VALUENE_BOOL, 
	// 	// 	CM, BOOL_ZOMBIE, NULL));

		GrB_OK(GrB_Matrix_select_BOOL(CDM, NULL, NULL, GrB_VALUEEQ_BOOL, 
			CM, BOOL_ZOMBIE, NULL));

	// 	// GrB_OK(GrB_Matrix_assign_BOOL(
	// 	// 	CDM, CDM, NULL, true, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_RS));
	}

	GrB_free(&accum);
	GrB_free(&maybe_del);

	return GrB_SUCCESS;
}

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
