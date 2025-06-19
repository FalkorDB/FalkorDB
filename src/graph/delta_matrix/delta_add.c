/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"

GrB_Info Delta_eWiseAdd                // C = A + B
(
    Delta_Matrix C,               // input/output matrix for results
    const GrB_Semiring semiring,  // defines '+' for T=A+B
    const Delta_Matrix A,         // first input:  matrix A
    const Delta_Matrix B          // second input: matrix B
) {
	ASSERT(A        != NULL);
	ASSERT(B        != NULL);
	ASSERT(C        != NULL);
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

