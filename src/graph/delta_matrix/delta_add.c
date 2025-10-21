/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_utils.h"
#include "delta_matrix.h"
#include "../../util/rmalloc.h"
#include "../../globals.h"


// Structurally add two delta matricies. Will output a boolean matrix. 
// C = A + B
GrB_Info Delta_add
(
    Delta_Matrix C,         // input/output matrix for results
    const Delta_Matrix A,   // first input:  matrix A
    const Delta_Matrix B    // second input: matrix B
) {
	Delta_Matrix_addCompatible(C, A, B);
	Delta_Matrix_validate(A, false);
	Delta_Matrix_validate(B, false);

	GrB_Type a_ty;
	GrB_Type b_ty;
	GrB_Type c_ty;
	Delta_Matrix_type(&a_ty, A);
	Delta_Matrix_type(&b_ty, B);
	Delta_Matrix_type(&c_ty, C);

	ASSERT(c_ty == GrB_BOOL);
	ASSERT(a_ty == GrB_BOOL || a_ty == GrB_UINT64);
	ASSERT(b_ty == GrB_BOOL || b_ty == GrB_UINT64);
	
	const struct GrB_ops *ops = Global_GrB_Ops_Get();
	GrB_Scalar alpha = a_ty == GrB_BOOL ? ops->bool_zombie : ops->u64_zombie;
	GrB_Scalar beta  = b_ty == GrB_BOOL ? ops->bool_zombie : ops->u64_zombie;
	Delta_eWiseUnion(C, GrB_ONEB_BOOL, A, alpha, B, beta);
	
	GrB_Matrix CM  = DELTA_MATRIX_M(C);
	GrB_Matrix CDM = DELTA_MATRIX_DELTA_MINUS(C);
	GrB_Index deletions = 0;
	GrB_Matrix_nvals(&deletions, CDM);

	// add zombies to removed entries in M
	if(deletions > 0){
		GrB_OK (GrB_Matrix_assign_BOOL(CM, CDM, NULL, BOOL_ZOMBIE, GrB_ALL, 0,
			GrB_ALL, 0, GrB_DESC_S));
	}

	GrB_Matrix_wait(CM, GrB_MATERIALIZE);
	Delta_Matrix_validate(C, false);

	return GrB_SUCCESS;
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
	Delta_Matrix_addCompatible(C, A, B);
	Delta_Matrix_validate(A, false);
	Delta_Matrix_validate(B, false);

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

	GrB_OK (GrB_Matrix_nvals(&adp_vals, ADP));
	GrB_OK (GrB_Matrix_nvals(&bdp_vals, BDP));
	GrB_OK (GrB_Matrix_nvals(&adm_vals, ADM));
	GrB_OK (GrB_Matrix_nvals(&bdm_vals, BDM));

	bool handle_addition = adp_vals || bdp_vals;
	bool handle_deletion = adm_vals || bdm_vals;
	
	if (!handle_addition){
		GrB_OK (GrB_Matrix_clear(CDP));
	}

	if (!handle_deletion && CDM != NULL) {
		GrB_OK (GrB_Matrix_clear(CDM));
	}

	GrB_OK (Delta_Matrix_nrows(&nrows, C));
	GrB_OK (Delta_Matrix_ncols(&ncols, C));
	GrB_OK (Delta_Matrix_type (&c_ty, C));

	//--------------------------------------------------------------------------
	// DM_union = (ADM - BM) ∪ (BDM - AM)
	//--------------------------------------------------------------------------
	if(handle_deletion) {
		GrB_OK (GrB_Matrix_new(&DM_union, GrB_BOOL, nrows, ncols));
		GrB_OK (GxB_set(DM_union, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE));

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
	// CM <CM> = AM + BDP equivalently CM <CM> = CM + BDP
	// CM <CM> = ADP + BM equivalently CM <CM> = ADP + CM
	//--------------------------------------------------------------------------

	// This code should call ewiseadd, but must wait for a certain GraphBLAS
	// kernel to be efficient.

	if(handle_addition) {
		GrB_OK (GrB_Matrix_new(&M_times_DP, c_ty, nrows, ncols));

		// Set M_times_DP to hypersparse since it is a submatrix of DP
		// so it shouldn't be CSR until we sync it.
		GrB_OK (GxB_set(M_times_DP, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE));


		GrB_OK (GrB_Matrix_eWiseMult_BinaryOp(M_times_DP, NULL, NULL, op, CM, 
			BDP, NULL));
		GrB_OK (GrB_Matrix_eWiseMult_BinaryOp(M_times_DP, M_times_DP, NULL, op, 
			ADP, CM, GrB_DESC_SC));

		// cannot use an accumulator since op might be an indexBinaryOp
		GrB_OK (GrB_Matrix_assign(CM, M_times_DP, NULL, M_times_DP, GrB_ALL, 
			nrows, GrB_ALL, ncols, GrB_DESC_S));
	}
	

	//--------------------------------------------------------------------------
	// CDP <!CM> = ADP + BDP 
	//--------------------------------------------------------------------------
	if (handle_addition){
		GrB_OK (GrB_Matrix_eWiseAdd_BinaryOp(CDP, M_times_DP, NULL, op, ADP, 
			BDP, GrB_DESC_SC));
	} else {
		GrB_OK (GrB_Matrix_clear(CDP));
	}

	// don't use again, could have been overwritten.
	ADP = BDP = NULL;

	//--------------------------------------------------------------------------
    // CDM <!CDP>= (ADM - BM) ∪ (BDM - AM) ∪ (ADM ∩ BDM)
	//--------------------------------------------------------------------------
	if(handle_deletion) {
		GrB_OK (GrB_Matrix_eWiseMult_BinaryOp(
			CDM, NULL, NULL, GrB_ONEB_BOOL, ADM, BDM, NULL));
	} else if (CDM != NULL) {
		GrB_OK (GrB_Matrix_clear(CDM));
	}

	// don't use again, could have been overwritten.
	ADM = BDM = NULL;

	if(handle_deletion){
		// delete intersection of CDM with CDP
		GrB_OK (GrB_Matrix_assign(CDM, M_times_DP, GrB_ONEB_BOOL, DM_union,
			GrB_ALL, 0, GrB_ALL, 0, handle_addition ? GrB_DESC_SC : NULL));
	}

	GrB_OK (GrB_Matrix_wait(CM, GrB_MATERIALIZE));
	GrB_free(&DM_union);
	GrB_free(&M_times_DP);
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
	Delta_Matrix_addCompatible(C, A, B);
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

	GrB_OK (GrB_Matrix_nvals(&adp_vals, ADP)) ;
	GrB_OK (GrB_Matrix_nvals(&adm_vals, ADM)) ;
	GrB_OK (GrB_Matrix_nvals(&bdp_vals, BDP)) ;
	GrB_OK (GrB_Matrix_nvals(&bdm_vals, BDM)) ;
	GrB_OK (Delta_Matrix_type (&C_ty, C)) ;
	GrB_OK (Delta_Matrix_nrows(&nrows, C)) ;
	GrB_OK (Delta_Matrix_ncols(&ncols, C)) ;

	#if RG_DEBUG
	GrB_OK (GrB_Matrix_apply_BinaryOp1st_Scalar(ADM, ADM, GrB_SECOND_BOOL, 
		GrB_EQ_UINT64, alpha, AM, GrB_DESC_S));
	GrB_OK (GrB_Matrix_apply_BinaryOp1st_Scalar(BDM, BDM, GrB_SECOND_BOOL, 
		GrB_EQ_UINT64, beta, BM, GrB_DESC_S));
	GrB_OK (GrB_Matrix_set_INT32(ADM, true, GxB_ISO));
	GrB_OK (GrB_Matrix_set_INT32(BDM, true, GxB_ISO));
	bool ok = true;
	GrB_OK (GrB_Matrix_reduce_BOOL(&ok, NULL, GrB_LAND_MONOID_BOOL, ADM, NULL));
	ASSERT(ok);
	GrB_OK (GrB_Matrix_reduce_BOOL(&ok, NULL, GrB_LAND_MONOID_BOOL, BDM, NULL));
	ASSERT(ok);
	#endif

	bool handle_deletion = adm_vals || bdm_vals;
	bool handle_addition = adp_vals || bdp_vals;


	//--------------------------------------------------------------------------
	// DM_union = (ADM - BM) ∪ (BDM - AM) 
	//--------------------------------------------------------------------------
	if(handle_deletion) {
		GrB_OK (GrB_Matrix_new(&DM_union, GrB_BOOL, nrows, ncols));
		GrB_OK (GrB_transpose(DM_union, BM, NULL, ADM, GrB_DESC_SCT0));
		GrB_OK (GrB_transpose(DM_union, AM, GrB_ONEB_BOOL, BDM, GrB_DESC_SCT0));
	}

	//--------------------------------------------------------------------------
	// M_times_DP =  AM * BDP U BM * ADP 
	//--------------------------------------------------------------------------
	if(handle_addition) {
		GrB_OK (GrB_Matrix_new(&M_times_DP, C_ty, nrows, ncols));

		// Set M_times_DP to hypersparse since it is a submatrix of DP
		// so it shouldn't be CSR until we sync it.
		GrB_OK (GrB_set(M_times_DP, GxB_HYPERSPARSE, GxB_SPARSITY_CONTROL));

		GrB_OK (GrB_Matrix_eWiseMult_BinaryOp(
			M_times_DP, NULL, NULL, op, AM, BDP, NULL));
			
		// The accum op is not used 
		// (since there should be no intersection between AM and BM)
		// However it is needed so that previous entries are not deleted. 
		GrB_OK (GrB_Matrix_eWiseMult_BinaryOp(
			M_times_DP, M_times_DP, NULL, op, BM, ADP, GrB_DESC_SC));
	}

	//--------------------------------------------------------------------------
	// M: CM = AM + BM ----- The bulk of the work.
	//--------------------------------------------------------------------------
	GrB_OK (GxB_Matrix_eWiseUnion(
		CM, NULL, NULL, op, AM, alpha, BM, beta, NULL));

	// don't use again, could have been overwritten.
	AM = BM = NULL;

	//--------------------------------------------------------------------------
	// CDP = ADP + BDP
	//--------------------------------------------------------------------------
	if(handle_addition) {
		GrB_OK (GxB_Matrix_eWiseUnion(
			CDP, M_times_DP, NULL, op, ADP, alpha, BDP, beta, GrB_DESC_SC));
	} else {
		GrB_OK (GrB_Matrix_clear(CDP));
	}

	// don't use again, could have been overwritten.
	ADP = BDP = NULL;

	//--------------------------------------------------------------------------
    // CDM <!CDP>= (ADM - BM) ∪ (BDM - AM) ∪ (ADM ∩ BDM)
	//--------------------------------------------------------------------------
	if(handle_deletion) {
		GrB_OK (GrB_Matrix_eWiseMult_BinaryOp(
			CDM, NULL, NULL, GrB_ONEB_BOOL, ADM, BDM, NULL));
	} else if (CDM != NULL) {
		GrB_OK (GrB_Matrix_clear(CDM));
	}
	// don't use again, could have been overwritten.
	ADM = BDM = NULL;

	if(handle_deletion){
		// delete intersection of CDM with CDP
		GrB_OK (GrB_Matrix_assign(CDM, M_times_DP, GrB_ONEB_BOOL, DM_union,
			GrB_ALL, 0, GrB_ALL, 0, handle_addition ? GrB_DESC_SC : NULL));
	}

	//--------------------------------------------------------------------------
	// CM <CM> = M_times_DP 
	//--------------------------------------------------------------------------
	if(M_times_DP) {
		GrB_OK (GrB_Matrix_assign(CM, M_times_DP, NULL, M_times_DP, GrB_ALL, 
			nrows, GrB_ALL, ncols, GrB_DESC_S));
	}

	GrB_OK (GrB_Matrix_wait(CM, GrB_MATERIALIZE));
	GrB_free(&DM_union);
	GrB_free(&M_times_DP);
	return GrB_SUCCESS;
}
