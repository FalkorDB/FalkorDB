/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_utils.h"
#include "delta_matrix.h"
#include "../../util/rmalloc.h"

// GrB_info _submatrix_assign
// (
// 	GrB_Matrix C,
// 	const GrB_Matrix A
// ) {
// 	GrB_Index nrows;
// 	GrB_Index ncols;
// 	GrB_Matrix_nrows(&nrows, A);
// 	GrB_Matrix_ncols(&ncols, A);
// 	GrB_Index *row_arr = rm_malloc(nrows * sizeof(GrB_Index));
// 	GrB_Index *col_arr = rm_malloc(ncols * sizeof(GrB_Index));


// }

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

	GrB_Matrix   CM         = DELTA_MATRIX_M(C);
	GrB_Matrix   CDP        = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix   CDM        = DELTA_MATRIX_DELTA_MINUS(C);
	GrB_Matrix   AM         = DELTA_MATRIX_M(A);
	GrB_Matrix   BM         = DELTA_MATRIX_M(B);
	GrB_Matrix   ADP        = DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix   ADM        = DELTA_MATRIX_DELTA_MINUS(A);
	GrB_Matrix   BDP        = DELTA_MATRIX_DELTA_PLUS(B);
	GrB_Matrix   BDM        = DELTA_MATRIX_DELTA_MINUS(B);
	GrB_Matrix   DM_union   = NULL;

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

	//--------------------------------------------------------------------------
	// DM_union = (ADM - BM) ∪ (BDM - AM)
	//--------------------------------------------------------------------------
	if(handle_deletion) {
		GrB_Matrix_new(&DM_union, GrB_BOOL, nrows, ncols);

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
		GrB_OK (GrB_transpose(CM, CM, op, CDP, GrB_DESC_ST0));
	}

	//--------------------------------------------------------------------------
	// CDP<!CM> = ADP + BDP ---- remove intersection with M
	//--------------------------------------------------------------------------
	if(handle_addition){
		GrB_OK (GrB_transpose(CDP, CM, NULL, CDP, GrB_DESC_RSCT0));
	}

	Delta_Matrix_wait(C, false);
	GrB_free(&DM_union);
	return GrB_SUCCESS;
}

// All zombies should be equal to alpha if in AM or beta if in BM
// C = A + B
// This is calculated by:
// CM        = AM + BM
// CM   <CM>+= ADP + BDP
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

	GrB_Matrix   CM         = DELTA_MATRIX_M(C);
	GrB_Matrix   CDP        = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix   CDM        = DELTA_MATRIX_DELTA_MINUS(C);
	GrB_Matrix   AM         = DELTA_MATRIX_M(A);
	GrB_Matrix   BM         = DELTA_MATRIX_M(B);
	GrB_Matrix   ADP        = DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix   ADM        = DELTA_MATRIX_DELTA_MINUS(A);
	GrB_Matrix   BDP        = DELTA_MATRIX_DELTA_PLUS(B);
	GrB_Matrix   BDM        = DELTA_MATRIX_DELTA_MINUS(B);
	GrB_Matrix   DM_union   = NULL;
	GrB_Matrix   M_times_DP = NULL;

	GrB_Matrix_nvals(&adp_vals, ADP);
	GrB_Matrix_nvals(&adm_vals, ADM);
	GrB_Matrix_nvals(&bdp_vals, BDP);
	GrB_Matrix_nvals(&bdm_vals, BDM);
	info = Delta_Matrix_type (&C_ty, C);
	ASSERT(info == GrB_SUCCESS);

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
		info = GrB_Matrix_assign(
			CM, M_times_DP, NULL, M_times_DP, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S);
		ASSERT(info == GrB_SUCCESS);
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
	return info;
}
