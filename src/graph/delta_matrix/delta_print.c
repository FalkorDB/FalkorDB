/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"

// print and check a GrB_Matrix
GrB_Info Delta_Matrix_fprint
(
    Delta_Matrix A,  // object to print and check
    int pr,          // print level (GxB_Print_Level)
    FILE *f          // file for output
) {
	ASSERT (A != NULL) ;
	ASSERT (f != NULL) ;

	if (DELTA_MATRIX_MAINTAIN_TRANSPOSE (A)) {
		Delta_Matrix AT = A->transposed ;

		GrB_Matrix mt  = Delta_Matrix_M  (AT) ;
		GrB_Matrix dpt = Delta_Matrix_DP (AT) ;
		GrB_Matrix dmt = Delta_Matrix_DM (AT) ;

		GrB_RETURN_IF_FAIL (GxB_Matrix_fprint (mt,  "MT",  pr, f))
		GrB_RETURN_IF_FAIL (GxB_Matrix_fprint (dpt, "DPT", pr, f))
		GrB_RETURN_IF_FAIL (GxB_Matrix_fprint (dmt, "DMT", pr, f))
	}

	GrB_Matrix m  = Delta_Matrix_M  (A) ;
	GrB_Matrix dp = Delta_Matrix_DP (A) ;
	GrB_Matrix dm = Delta_Matrix_DM (A) ;

	GrB_RETURN_IF_FAIL (GxB_Matrix_fprint (m,  "M",  pr, f))
	GrB_RETURN_IF_FAIL (GxB_Matrix_fprint (dp, "DP", pr, f))
	GrB_RETURN_IF_FAIL (GxB_Matrix_fprint (dm, "DM", pr, f))

	return GrB_SUCCESS ;
}

