/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"

// zombies should be the monoid's identity value.
// C = A + B
// CDM =
GrB_Info Delta_eWiseAdd
(
    Delta_Matrix C,       // input/output matrix for results
    const GrB_Monoid op,  // defines '+' for T=A+B
    const Delta_Matrix A, // first input:  matrix A
    const Delta_Matrix B  // second input: matrix B
) {
	ASSERT(A  != NULL);
	ASSERT(B  != NULL);
	ASSERT(C  != NULL);
	ASSERT(op != NULL);


	GrB_Info  info;
	GrB_Index nrows;
	GrB_Index ncols;

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
	GrB_Matrix   dm_and_dp  = NULL;
	GrB_BinaryOp biop       = NULL;

	Delta_Matrix_nrows(&nrows, C);
	Delta_Matrix_ncols(&ncols, C);
	GrB_Monoid_get_VOID(op, (void *) &biop, GxB_MONOID_OPERATOR);

	GrB_Matrix_new(&DM_union, GrB_BOOL, nrows, ncols);
	GxB_Global_Option_set(GxB_BURBLE, true);
	
	//--------------------------------------------------------------------------
	// DM_union = ADM ∩ BDM 
	//--------------------------------------------------------------------------
	info = GrB_Matrix_assign(
		DM_union, BM, NULL, ADM, GrB_ALL, 0, GrB_ALL, 0 , GrB_DESC_SC);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_assign(
		DM_union, AM, GrB_ONEB_BOOL, BDM, GrB_ALL, 0, GrB_ALL, 0 , GrB_DESC_SC);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// M: CM = AM + BM ----- The bulk of the work.
	//--------------------------------------------------------------------------
	info = GrB_Matrix_eWiseAdd_Monoid(CM, NULL, NULL, op, AM, BM, NULL);
	ASSERT(info == GrB_SUCCESS);
	// don't use again, could have been overwritten.
	AM = BM = NULL;

	//--------------------------------------------------------------------------
	// CDP = ADP + BDP ---- Must later remove intersection with M
	//--------------------------------------------------------------------------
	info = GrB_Matrix_eWiseAdd_Monoid(CDP, NULL, NULL, op, ADP, BDP, NULL);
	ASSERT(info == GrB_SUCCESS);

	// don't use again, could have been overwritten.
	ADP = BDP = NULL;

	//--------------------------------------------------------------------------
	// CDM = (ADM ∩ BDM) ∪ ((dmA ∪ dmB ) - (CM ∪ CDP))
	//--------------------------------------------------------------------------
	info = GrB_Matrix_eWiseMult_BinaryOp(
		CDM, NULL, NULL, GrB_ONEB_BOOL, ADM, BDM, NULL);
	ASSERT(info == GrB_SUCCESS);

	// don't use again, could have been overwritten.
	ADM = BDM = NULL;

	info = GrB_transpose(
		CDM, CDP, GrB_ONEB_BOOL, DM_union, GrB_DESC_SCT0);
	ASSERT(info == GrB_SUCCESS); 

	//--------------------------------------------------------------------------
	// CM <CM>+= CDP ----- Should be done inplace (currently is not GBLAS TODO)
	//--------------------------------------------------------------------------
	info = GrB_Matrix_assign(
		CM, CM, biop, CDP, GrB_ALL, 0, GrB_ALL, 0 , GrB_DESC_S);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// CDP<!CM> = ADP + BDP ---- remove intersection with M
	//--------------------------------------------------------------------------
	info = GrB_Matrix_assign(
		CDP, CM, NULL, CDP, GrB_ALL, 0, GrB_ALL, 0 , GrB_DESC_RSC);
	ASSERT(info == GrB_SUCCESS);
	GxB_Global_Option_set(GxB_BURBLE, false);

	Delta_Matrix_wait(C, false);
	GrB_free(&DM_union);
	return info;
}
