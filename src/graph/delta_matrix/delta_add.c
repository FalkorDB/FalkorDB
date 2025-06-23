/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"

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

	// TODO: properly handle alliasing 
	// ASSERT(C != A);  // sketchy 
	ASSERT(C != B); // will break
	ASSERT (Delta_Matrix_Synced(C));

	GrB_Info  info;
	GrB_Index nrows;
	GrB_Index ncols;
	GrB_Index ADM_nvals;
	GrB_Index BDM_nvals;
	GrB_Index DP_nvals;

	GrB_Matrix   _C   = DELTA_MATRIX_M(C);
	GrB_Matrix   AM   = DELTA_MATRIX_M(A);
	GrB_Matrix   BM   = DELTA_MATRIX_M(B);
	GrB_Matrix   ADP  = DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix   ADM  = DELTA_MATRIX_DELTA_MINUS(A);
	GrB_Matrix   BDP  = DELTA_MATRIX_DELTA_PLUS(B);
	GrB_Matrix   BDM  = DELTA_MATRIX_DELTA_MINUS(B);
	GrB_BinaryOp biop = NULL;


	GrB_Matrix_nvals(&ADM_nvals, ADM);
	GrB_Matrix_nvals(&BDM_nvals, BDM);

	// need biop of 
	info = GrB_Monoid_get_VOID (op, (void *) &biop, GxB_MONOID_OPERATOR);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// C = A + B
	//--------------------------------------------------------------------------
	
	if(ADM_nvals > 0 || BDM_nvals > 0) { 
		// if AM && BM && BDM, then _C gets the entry of AM.
		// this is accomplised by the mask and accumulator. 
		// ewiseadd would not work.
		if(_C != A){
			info = GrB_transpose(_C, ADM, NULL, AM, GrB_DESC_SCT0);
			ASSERT(info == GrB_SUCCESS);
		}
		info = GrB_transpose(_C, BDM, biop, BM, GrB_DESC_SCT0);
	} else {
		info = GrB_Matrix_eWiseAdd_Monoid(_C, NULL, NULL, op, AM, BM, NULL);
	}
	ASSERT(info == GrB_SUCCESS);

	// Add the changes from the delta plus matricies.
	info = GrB_Matrix_eWiseAdd_Monoid(_C, NULL, NULL, op, _C, ADP, NULL);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_eWiseAdd_Monoid(_C, NULL, NULL, op, _C, BDP, NULL);
	ASSERT(info == GrB_SUCCESS);

	return info;
}

