/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"
#include "delta_utils.h"

// Set the internal matricies of C
// the operation can only succeed if C's internal matrices are all empty
GrB_Info Delta_Matrix_setMatrices
(
	Delta_Matrix C,  // delta matrix
	GrB_Matrix *M,   // new M
	GrB_Matrix *DP,  // new delta-plus
	GrB_Matrix *DM   // new delta-minus
) {
	ASSERT (C   != NULL) ;
	ASSERT (M   != NULL) ;
	ASSERT (DP  != NULL) ;
	ASSERT (DM  != NULL) ;
	ASSERT (*M  != NULL) ;
	ASSERT (*DP != NULL) ;
	ASSERT (*DM != NULL) ;

	GrB_Index nvals = 0;

	// verify that C is empty
	ASSERT (Delta_Matrix_Synced (C)) ;
	Delta_Matrix_nvals (&nvals, C) ;
	ASSERT (nvals == 0) ;

	GrB_OK (GrB_free (&DELTA_MATRIX_M (C))) ;
	GrB_OK (GrB_free (&DELTA_MATRIX_DELTA_PLUS (C))) ;
	GrB_OK (GrB_free (&DELTA_MATRIX_DELTA_MINUS (C))) ;

	DELTA_MATRIX_M(C)           = *M ;
	DELTA_MATRIX_DELTA_PLUS(C)  = *DP ;
	DELTA_MATRIX_DELTA_MINUS(C) = *DM ;
	GrB_OK (GrB_Matrix_wait (*M, GrB_MATERIALIZE)) ;

	// set correct sparcity controls
	GrB_OK (GrB_set (*M,  GxB_SPARSE | GxB_HYPERSPARSE, GxB_SPARSITY_CONTROL)) ;
	GrB_OK (GrB_set (*DP, GxB_HYPERSPARSE, GxB_SPARSITY_CONTROL)) ;
	GrB_OK (GrB_set (*DM, GxB_HYPERSPARSE, GxB_SPARSITY_CONTROL)) ;

	*M  = NULL ;
	*DP = NULL ;
	*DM = NULL ;

	Delta_Matrix_validate (C, false) ;

	return GrB_SUCCESS ;
}

// set the internal matrix M
// the operation can only succeed if C's internal matrices are all empty
GrB_Info Delta_Matrix_setM
(
	Delta_Matrix C,  // delta matrix
	GrB_Matrix *M    // new M
) {
	GrB_Index nvals = 0;
	GrB_Index tot   = 0;
	GrB_OK (GrB_Matrix_nvals(&nvals, DELTA_MATRIX_M(C)));
	tot |= nvals;

	GrB_OK (GrB_Matrix_nvals(&nvals, DELTA_MATRIX_DELTA_PLUS(C)));
	tot |= nvals;

	GrB_OK (GrB_Matrix_nvals(&nvals, DELTA_MATRIX_DELTA_MINUS(C)));
	tot |= nvals;

	if (tot != 0)
		return GrB_ALREADY_SET;

	GrB_OK (GrB_free(&DELTA_MATRIX_M(C)));

	DELTA_MATRIX_M(C) = *M;

	GrB_OK (GrB_Matrix_wait(*M, GrB_MATERIALIZE));

	// set correct sparcity controls
	GrB_OK (GrB_set(*M, GxB_SPARSE | GxB_HYPERSPARSE, GxB_SPARSITY_CONTROL));

	*M = NULL;

	return GrB_SUCCESS;
}

// get the internal matrix M
GrB_Matrix Delta_Matrix_M
(
	const Delta_Matrix C  // delta matrix
) {
	return DELTA_MATRIX_M(C);
}

// get the internal matrix delta plus
GrB_Matrix Delta_Matrix_DP
(
	const Delta_Matrix C  // delta matrix
) {
	return DELTA_MATRIX_DELTA_PLUS(C);
}

// get the internal matrix delta minus
GrB_Matrix Delta_Matrix_DM
(
	const Delta_Matrix C  // delta matrix
) {
	return DELTA_MATRIX_DELTA_MINUS(C);
}
