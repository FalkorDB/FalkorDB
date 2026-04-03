/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"
#include "delta_utils.h"

GrB_Info Delta_eWiseAdd                // C = A + B
(
    Delta_Matrix C,               // input/output matrix for results
    const GrB_Semiring semiring,  // defines '+' for T=A+B
    const Delta_Matrix A,         // first input:  matrix A
    const Delta_Matrix B          // second input: matrix B
) {
	Delta_Matrix_addCompatible(C, A, B);
	ASSERT(semiring != NULL);

	GrB_Index  nrows;
	GrB_Index  ncols;
	GrB_Index a_rows;
	GrB_Index a_cols;
	GrB_Index b_rows;
	GrB_Index b_cols;

	GrB_Index  DM_nvals;
	GrB_Index  DP_nvals;

	GrB_Matrix _A  = NULL;
	GrB_Matrix _B  = NULL;
	GrB_Matrix _C  = DELTA_MATRIX_M(C);
	GrB_Matrix AM  = DELTA_MATRIX_M(A);
	GrB_Matrix BM  = DELTA_MATRIX_M(B);
	GrB_Matrix ADP = DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix ADM = DELTA_MATRIX_DELTA_MINUS(A);
	GrB_Matrix BDP = DELTA_MATRIX_DELTA_PLUS(B);
	GrB_Matrix BDM = DELTA_MATRIX_DELTA_MINUS(B);

	Delta_Matrix_nrows(&nrows, C);
	Delta_Matrix_ncols(&ncols, C);

	if(Delta_Matrix_Synced(A)) {
		_A = AM;
	} else {
		GrB_OK (Delta_Matrix_export(&_A, A, GrB_BOOL));
	}

	GrB_OK (GrB_Matrix_nvals(&DM_nvals, BDM));
	GrB_OK (GrB_Matrix_nvals(&DP_nvals, BDP));

	if(Delta_Matrix_Synced(B)) {
		_B = BM;
	} else {
		GrB_OK (Delta_Matrix_export(&_B, B, GrB_BOOL));
	}

	//--------------------------------------------------------------------------
	// C = A + B
	//--------------------------------------------------------------------------

	GrB_OK (GrB_Matrix_eWiseAdd_Semiring(_C, NULL, NULL, semiring, _A, _B,
		NULL));

	Delta_Matrix_wait(C, false);

	if(_A != AM) GrB_free(&_A);
	if(_B != BM) GrB_free(&_B);

	return GrB_SUCCESS;
}

