/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"

// zombies must be the identity of the given monoid.
// C = A + B
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
	GrB_Index ADM_nvals;
	GrB_Index BDM_nvals;
	GrB_Index DP_nvals;

	GrB_Matrix   CM         = DELTA_MATRIX_M(C);
	GrB_Matrix   CDP        = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix   CDM        = DELTA_MATRIX_DELTA_MINUS(C);
	GrB_Matrix   AM         = DELTA_MATRIX_M(A);
	GrB_Matrix   BM         = DELTA_MATRIX_M(B);
	GrB_Matrix   ADP        = DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix   ADM        = DELTA_MATRIX_DELTA_MINUS(A);
	GrB_Matrix   BDP        = DELTA_MATRIX_DELTA_PLUS(B);
	GrB_Matrix   BDM        = DELTA_MATRIX_DELTA_MINUS(B);
	GrB_Matrix   dm_and_dp  = NULL;
	GrB_BinaryOp biop       = NULL;


	GrB_Matrix_nvals(&ADM_nvals, ADM);
	GrB_Matrix_nvals(&BDM_nvals, BDM);

	//--------------------------------------------------------------------------
	// M: CM = AM + BM ----- The bulk of the work.
	//--------------------------------------------------------------------------
	info = GrB_Matrix_eWiseAdd_Monoid(CM, NULL, NULL, op, AM, BM, NULL);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// DP: CDP = ADP + BDP ---- Must later remove intersection with M
	//--------------------------------------------------------------------------
	info = GrB_Matrix_eWiseAdd_Monoid(CDP, NULL, NULL, op, ADP, BDP, NULL);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// DM: CDM<!DP> = ADM + BDM 
	//--------------------------------------------------------------------------
	info = GrB_Matrix_eWiseAdd_Monoid(
		CDM, CDP, NULL, op, ADM, BDM, GrB_DESC_SC);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// DP: CDP<!CM> = ADP + BDP ---- remove intersection with M
	//--------------------------------------------------------------------------
	info = GrB_Matrix_assign(CDP, CM, NULL, CDP, GrB_ALL, 0, GrB_ALL, 0 , GrB_DESC_SC);
	ASSERT(info == GrB_SUCCESS);

	Delta_Matrix_wait(C, false);
	return info;
}
