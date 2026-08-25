/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"

GrB_Info Delta_transpose_calculate
(
	Delta_Matrix C
) {
	ASSERT (C != NULL) ;

	GrB_Index nrows = 0 ;
	GrB_Index ncols = 0 ;

	GrB_OK (Delta_Matrix_nrows (&nrows, C)) ;
	GrB_OK (Delta_Matrix_ncols (&ncols, C)) ;

	Delta_Matrix CT = Delta_Matrix_getTranspose (C) ;
	if (CT == NULL) {
		GrB_OK (Delta_Matrix_new (&CT, GrB_BOOL, ncols, nrows, false)) ;

		C->transposed = CT ;
	} else {
		GrB_OK (Delta_Matrix_clear (CT)) ;
		GrB_OK (Delta_Matrix_resize (CT, ncols, nrows)) ;
	}

	GrB_Matrix M   = DELTA_MATRIX_M           (C) ;
	GrB_Matrix DP  = DELTA_MATRIX_DELTA_PLUS  (C) ;
	GrB_Matrix DM  = DELTA_MATRIX_DELTA_MINUS (C) ;
	GrB_Matrix TM  = DELTA_MATRIX_M           (CT) ;
	GrB_Matrix TDP = DELTA_MATRIX_DELTA_PLUS  (CT) ;
	GrB_Matrix TDM = DELTA_MATRIX_DELTA_MINUS (CT) ;

	GrB_OK (GrB_apply (TM, NULL, NULL, GxB_ONE_BOOL, M, GrB_DESC_T0)) ;
	GrB_OK (GrB_apply (TDP, NULL, NULL, GxB_ONE_BOOL, DP, GrB_DESC_T0)) ;
	GrB_OK (GrB_apply (TDM, NULL, NULL, GxB_ONE_BOOL, DM, GrB_DESC_T0)) ;

	return GrB_SUCCESS ;
}

