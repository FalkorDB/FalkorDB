/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_utils.h"
#include "delta_matrix.h"
#include "../../util/rmalloc.h"

// preforms C <A>= A. Assuming A is a submatrix of C, this should be done in  
// place. TODO does assign do this inplace?
void _assign_inplace
(
	GrB_Matrix C,
	const GrB_Matrix A
) {
	GrB_Index nrows;
	GrB_Index ncols;
	GrB_Index r;
	GrB_Index c;
	GrB_Type  ty;
	uint64_t v_int;
	bool     v_bool;

	GrB_Matrix_nrows(&nrows, A);
	GrB_Matrix_ncols(&ncols, A);
	GxB_Matrix_type (&ty, A);
	ASSERT(ty == GrB_UINT64 || ty == GrB_BOOL);
	
	struct GB_Iterator_opaque _i;
	GxB_Iterator i = &_i;
	GrB_OK(GxB_Matrix_Iterator_attach(i, A, NULL));
	
	GrB_Info it_info = GxB_Matrix_Iterator_seek(i, 0); 
	if(ty == GrB_UINT64){
		while(it_info == GrB_SUCCESS) {
			GxB_Matrix_Iterator_getIndex(i, &r, &c);
			v_int = GxB_Iterator_get_UINT64(i);			
			GrB_OK(GrB_Matrix_setElement_UINT64(C, v_int, r, c));
			it_info = GxB_Matrix_Iterator_next(i);
		}
	} else {	
		while(it_info == GrB_SUCCESS) {
			GxB_Matrix_Iterator_getIndex(i, &r, &c);
			v_int = GxB_Iterator_get_BOOL(i);			
			GrB_OK(GrB_Matrix_setElement_BOOL(C, v_int, r, c));
			it_info = GxB_Matrix_Iterator_next(i);
		}
	}
	ASSERT(it_info == GxB_EXHAUSTED);
}

// zombies should be the monoid's identity value.
// C = A + B
// This is calculated by:
// CM        = AM + BM
// CM   <CM>+= ADP + BDP
// CDP  <!CM>= ADP + BDP
// CDM <!CDP>= (ADM - BM) ∪ (BDM - AM) ∪ (ADM ∩ BDM)
GrB_Info Delta_eWiseAdd
(
    Delta_Matrix C,         // input/output matrix for results
    const GrB_BinaryOp op,  // defines '+' for T=A+B
    const Delta_Matrix A,   // first input:  matrix A
    const Delta_Matrix B    // second input: matrix B
) {
	ASSERT(A  != NULL);
	ASSERT(B  != NULL);
	ASSERT(C  != NULL);
	ASSERT(op != NULL);
	ASSERT(!DELTA_MATRIX_MAINTAIN_TRANSPOSE(C));

	GrB_Index nrows;
	GrB_Index ncols;
	GrB_Index adp_vals;
	GrB_Index adm_vals;
	GrB_Index bdp_vals;
	GrB_Index bdm_vals; 
	GrB_Type  c_ty;

	GrB_Matrix CM         = DELTA_MATRIX_M(C);
	GrB_Matrix CDP        = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix CDM        = DELTA_MATRIX_DELTA_MINUS(C);
	GrB_Matrix AM         = DELTA_MATRIX_M(A);
	GrB_Matrix BM         = DELTA_MATRIX_M(B);
	GrB_Matrix ADP        = DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix ADM        = DELTA_MATRIX_DELTA_MINUS(A);
	GrB_Matrix BDP        = DELTA_MATRIX_DELTA_PLUS(B);
	GrB_Matrix BDM        = DELTA_MATRIX_DELTA_MINUS(B);
	GrB_Matrix DM_union   = NULL;
	GrB_Matrix M_times_DP = NULL;

	// Set DM_union and M_times_DP union to be hypersparse like DP and DM
	
	GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);

	GrB_Matrix_nvals(&adp_vals, ADP);
	GrB_Matrix_nvals(&adm_vals, ADM);
	GrB_Matrix_nvals(&bdp_vals, BDP);
	GrB_Matrix_nvals(&bdm_vals, BDM);

	bool handle_deletion = adm_vals || bdm_vals;
	bool handle_addition = adp_vals || bdp_vals;
	
	if(!handle_addition && !handle_deletion) {
		GrB_Matrix_clear(CDM);
		GrB_Matrix_clear(CDP);
		return GrB_Matrix_eWiseAdd_BinaryOp(CM, NULL, NULL, op, AM, BM, NULL);
	}

	Delta_Matrix_nrows(&nrows, C);
	Delta_Matrix_ncols(&ncols, C);
	Delta_Matrix_type (&c_ty, C);

	//--------------------------------------------------------------------------
	// DM_union = (ADM - BM) ∪ (BDM - AM)
	//--------------------------------------------------------------------------
	if(handle_deletion) {
		GrB_Matrix_new(&DM_union, GrB_BOOL, nrows, ncols);
		GrB_OK(GxB_set(DM_union, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE));
		GrB_OK (GrB_transpose(DM_union, BM, NULL, ADM, GrB_DESC_SCT0));

		GrB_OK (GrB_transpose(
			DM_union, AM, GrB_ONEB_BOOL, BDM, GrB_DESC_SCT0));
	}

	//--------------------------------------------------------------------------
	// M: CM = AM + BM ----- The bulk of the work.
	//--------------------------------------------------------------------------
	GrB_OK (GrB_Matrix_eWiseAdd_BinaryOp(CM, NULL, NULL, op, AM, BM, NULL));

	// don't use again, could have been overwritten.
	AM = BM = NULL;

	//--------------------------------------------------------------------------
	// CDP = ADP + BDP ---- Must later remove intersection with M
	//--------------------------------------------------------------------------
	GrB_OK (GrB_Matrix_eWiseAdd_BinaryOp(CDP, NULL, NULL, op, ADP, BDP, NULL));

	// don't use again, could have been overwritten.
	ADP = BDP = NULL;

	//--------------------------------------------------------------------------
    // CDM <!CDP>= (ADM - BM) ∪ (BDM - AM) ∪ (ADM ∩ BDM)
	//--------------------------------------------------------------------------
	GrB_OK (GrB_Matrix_eWiseMult_BinaryOp(
		CDM, NULL, NULL, GrB_ONEB_BOOL, ADM, BDM, NULL));

	// don't use again, could have been overwritten.
	ADM = BDM = NULL;

	if(handle_deletion && handle_addition){
		// delete intersection of CDM with CDP
		GrB_OK (GrB_transpose(
			CDM, CDP, GrB_ONEB_BOOL, DM_union, GrB_DESC_SCT0));
	}

	//--------------------------------------------------------------------------
	// CM <CM>+= CDP ----- Should be done inplace (currently is not GBLAS TODO)
	//--------------------------------------------------------------------------

	// if the operator does not care which value it returns (CM or CDP) the next 
	// step is unnessecary.
	if(handle_addition && op != GrB_ONEB_BOOL && op != GxB_ANY_BOOL && 
		op != GrB_ONEB_UINT64 && op != GxB_ANY_UINT64)
	{
		// cannot use an accumulator since op might be an indexBinaryOp
		GrB_OK(GrB_Matrix_new(&M_times_DP, c_ty, nrows, ncols));
		GrB_OK(GxB_set(M_times_DP, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE));
		GrB_OK(GrB_Matrix_eWiseMult_BinaryOp(M_times_DP, NULL, NULL, op, CM, 
			CDP, NULL));

		// TODO: This might be faster in the future if we can assign in place.
		GrB_OK(GrB_Matrix_assign(CM, M_times_DP, NULL, M_times_DP, GrB_ALL, 
			nrows, GrB_ALL, ncols, GrB_DESC_S));
	}        

	//------ --------------------------------------------------------------------
	// CDP <!CM> = ADP + BDP ---- remove intersection with M
	//------ --------------------------------------------------------------------
	if(handle_addition){
		GrB_OK  (GrB_transpose(CDP, CM, NULL, CDP, GrB_DESC_RSCT0));
	}        

	Delta_Matrix_wait(C, false);
	GrB_free(&DM_union);
	GrB_free(&M_times_DP);
	GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);
	return GrB_SUCCESS;
}

// All zombies should be equal to alpha if in AM or beta if in BM
// C = A + B
// This is calculated by:
// CM        = AM + BM
// CM   <CM> = ADP*BM + BDP
// CDP  <!CM>= ADP + BDP
// CDM <!CDP>= (ADM - BM) ∪ (BDM - AM) ∪ (ADM ∩ BDM)
GrB_Info Delta_eWiseUnion
(
    Delta_Matrix C,          // input/output matrix for results
    const GrB_BinaryOp op,   // defines '+' for T=A+B
    const Delta_Matrix A,    // first input:  matrix A
	const GrB_Scalar alpha,  // second input: empty value of matrix A
    const Delta_Matrix B,    // three input: matrix B
	const GrB_Scalar beta    // fourth input: empty value of matrix B
) {
	ASSERT(A  != NULL);
	ASSERT(B  != NULL);
	ASSERT(C  != NULL);
	ASSERT(op != NULL);
	ASSERT(!DELTA_MATRIX_MAINTAIN_TRANSPOSE(C));

	GrB_Info  info;
	GrB_Index nrows;
	GrB_Index ncols;
	GrB_Index adp_vals;
	GrB_Index adm_vals;
	GrB_Index bdp_vals;
	GrB_Index bdm_vals;
	GrB_Type  C_ty;

	GrB_Matrix CM         = DELTA_MATRIX_M(C);
	GrB_Matrix CDP        = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix CDM        = DELTA_MATRIX_DELTA_MINUS(C);
	GrB_Matrix AM         = DELTA_MATRIX_M(A);
	GrB_Matrix BM         = DELTA_MATRIX_M(B);
	GrB_Matrix ADP        = DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix ADM        = DELTA_MATRIX_DELTA_MINUS(A);
	GrB_Matrix BDP        = DELTA_MATRIX_DELTA_PLUS(B);
	GrB_Matrix BDM        = DELTA_MATRIX_DELTA_MINUS(B);
	GrB_Matrix DM_union   = NULL;
	GrB_Matrix M_times_DP = NULL;

	GrB_Matrix_nvals(&adp_vals, ADP);
	GrB_Matrix_nvals(&adm_vals, ADM);
	GrB_Matrix_nvals(&bdp_vals, BDP);
	GrB_Matrix_nvals(&bdm_vals, BDM);
	info = Delta_Matrix_type (&C_ty, C);
	ASSERT(info == GrB_SUCCESS);
	GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);

	bool handle_deletion = adm_vals || bdm_vals;
	bool handle_addition = adp_vals || bdp_vals;

	if(!handle_addition && !handle_deletion) {
		GrB_Matrix_clear(CDM);
		GrB_Matrix_clear(CDP);
		return GxB_Matrix_eWiseUnion(
			CM, NULL, NULL, op, AM, alpha, BM, beta, NULL);
	}
	Delta_Matrix_nrows(&nrows, C);
	Delta_Matrix_ncols(&ncols, C);

	//--------------------------------------------------------------------------
    // DM_union = (ADM - BM) ∪ (BDM - AM) 
	//--------------------------------------------------------------------------
	if(handle_deletion) {
		GrB_Matrix_new(&DM_union, GrB_BOOL, nrows, ncols);

		info = GrB_transpose(DM_union, BM, NULL, ADM, GrB_DESC_SCT0);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_transpose(DM_union, AM, GrB_ONEB_BOOL, BDM, GrB_DESC_SCT0);
		ASSERT(info == GrB_SUCCESS);
	}

	//--------------------------------------------------------------------------
	// M_times_DP =  AM * BDP U BM * ADP 
	//--------------------------------------------------------------------------
	if(handle_addition && op != GrB_ONEB_BOOL) {
		GrB_Matrix_new(&M_times_DP, C_ty, nrows, ncols);

		info = GrB_Matrix_eWiseMult_BinaryOp(
			M_times_DP, NULL, NULL, op, AM, BDP, NULL);
		ASSERT(info == GrB_SUCCESS);
			
		// The accum op is not used 
		// (since there should be no intersection between AM and BM)
		// However it is needed so that previous entries are not deleted. 
		info = GrB_Matrix_eWiseMult_BinaryOp(
			M_times_DP, NULL, GrB_ONEB_UINT64, op, BM, ADP, NULL);
		ASSERT(info == GrB_SUCCESS);

		GxB_fprint(M_times_DP, 2, stdout);
	}

	//--------------------------------------------------------------------------
	// M: CM = AM + BM ----- The bulk of the work.
	//--------------------------------------------------------------------------
	info = GxB_Matrix_eWiseUnion(
		CM, NULL, NULL, op, AM, alpha, BM, beta, NULL);
	ASSERT(info == GrB_SUCCESS);

	// don't use again, could have been overwritten.
	AM = BM = NULL;

	//--------------------------------------------------------------------------
	// CDP = ADP + BDP ---- Must later remove intersection with M
	//--------------------------------------------------------------------------
	info = GxB_Matrix_eWiseUnion(
		CDP, NULL, NULL, op, ADP, alpha, BDP, beta, NULL);
	ASSERT(info == GrB_SUCCESS);

	// don't use again, could have been overwritten.
	ADP = BDP = NULL;

	//--------------------------------------------------------------------------
    // CDM <!CDP>= (ADM - BM) ∪ (BDM - AM) ∪ (ADM ∩ BDM)
	//--------------------------------------------------------------------------
	if(handle_deletion) {
		info = GrB_Matrix_eWiseMult_BinaryOp(
			CDM, NULL, NULL, GrB_ONEB_BOOL, ADM, BDM, NULL);
		ASSERT(info == GrB_SUCCESS);
	}

	// don't use again, could have been overwritten.
	ADM = BDM = NULL;

	if(handle_addition && handle_deletion) {
		info = GrB_transpose(
			CDM, CDP, GrB_ONEB_BOOL, DM_union, GrB_DESC_SCT0);
		ASSERT(info == GrB_SUCCESS); 
	}

	//--------------------------------------------------------------------------
	// CM <CM>= M_times_DP 
	//--------------------------------------------------------------------------
	if(M_times_DP) {
		// TODO: which one of these is faster?
		GrB_OK(GrB_Matrix_assign(CM, M_times_DP, NULL, M_times_DP, GrB_ALL, 
			nrows, GrB_ALL, ncols, GrB_DESC_S));
		// _assign_inplace(CM, M_times_DP);
	}

	//--------------------------------------------------------------------------
	// CDP<!CM> = ADP + BDP ---- remove intersection with M
	//--------------------------------------------------------------------------
	if(handle_addition) {
		info = GrB_transpose(
			CDP, CM, NULL, CDP, GrB_DESC_RSCT0);
		ASSERT(info == GrB_SUCCESS);
	}

	Delta_Matrix_wait(C, false);
	GrB_free(&DM_union);
	GrB_free(&M_times_DP);
	GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);
	return info;
}

// FIXME: remove. used for benchmarking.
GrB_Info Delta_eWiseAdd_OLD                // C = A + B
(
    Delta_Matrix C,                    // input/output matrix for results
    const GrB_Semiring semiring,    // defines '+' for T=A+B
    const Delta_Matrix A,              // first input:  matrix A
    const Delta_Matrix B               // second input: matrix B
) {
	ASSERT(A != NULL);
	ASSERT(B != NULL);
	ASSERT(C != NULL);
	ASSERT(semiring != NULL);

	GrB_Info        info;
	GrB_Index       nrows;
	GrB_Index       ncols;
	GrB_Index       DM_nvals;
	GrB_Index       DP_nvals;

	GrB_Matrix      _A    =  NULL;
	GrB_Matrix      _B    =  NULL;
	GrB_Matrix      _C    =  DELTA_MATRIX_M(C);
	GrB_Matrix      AM    =  DELTA_MATRIX_M(A);
	GrB_Matrix      BM    =  DELTA_MATRIX_M(B);
	GrB_Matrix      ADP   =  DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix      ADM   =  DELTA_MATRIX_DELTA_MINUS(A);
	GrB_Matrix      BDP   =  DELTA_MATRIX_DELTA_PLUS(B);
	GrB_Matrix      BDM   =  DELTA_MATRIX_DELTA_MINUS(B);

	// TODO: check A, B and C are compatible

	GrB_Matrix_nvals(&DM_nvals, ADM);
	GrB_Matrix_nvals(&DP_nvals, ADP);
	if(DM_nvals > 0 || DP_nvals > 0) {
		info = Delta_Matrix_export(&_A, A);
		ASSERT(info == GrB_SUCCESS);
	} else {
		_A = AM;
	}

	GrB_Matrix_nvals(&DM_nvals, BDM);
	GrB_Matrix_nvals(&DP_nvals, BDP);
	if(DM_nvals > 0 || DP_nvals > 0) {
		info = Delta_Matrix_export(&_B, B);
		ASSERT(info == GrB_SUCCESS);
	} else {
		_B = BM;
	}

	//--------------------------------------------------------------------------
	// C = A + B
	//--------------------------------------------------------------------------

	info = GrB_Matrix_eWiseAdd_Semiring(_C, NULL, NULL, semiring, _A, _B, NULL);
	ASSERT(info == GrB_SUCCESS);

	if(_A != AM) GrB_free(&_A);
	if(_B != BM) GrB_free(&_B);

	return info;
}
