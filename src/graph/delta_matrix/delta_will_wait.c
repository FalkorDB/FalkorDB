/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"

// checks if C or its transpose (if exists) will trigger a GrB_wait
// as a result of some pending work that GraphBLAS need to perform
GrB_Info Delta_Matrix_willWait
(
	const Delta_Matrix C,  // matrix to query
	bool *willWait         // [output] true if the matrix requires GrB_wait
) {
	ASSERT (C        != NULL) ;
	ASSERT (willWait != NULL) ;

	int32_t p   = false ;
	bool    res = false ;

	if (DELTA_MATRIX_MAINTAIN_TRANSPOSE (C)) {
		GrB_OK (Delta_Matrix_willWait (C->transposed, &res)) ;
		if (res == true) {
			*willWait = true ;
			return GrB_SUCCESS ;
		}
	}

	GrB_Matrix M  = DELTA_MATRIX_M           (C) ;
	GrB_Matrix DP = DELTA_MATRIX_DELTA_PLUS  (C) ;
	GrB_Matrix DM = DELTA_MATRIX_DELTA_MINUS (C) ;

	// check if M contains pending changes
	GrB_OK (GrB_Matrix_get_INT32 (M, &p, GxB_WILL_WAIT)) ;
	res = res || p == 1 ;

	// check if delta-plus contains pending changes
	GrB_OK (GrB_Matrix_get_INT32 (DP, &p, GxB_WILL_WAIT)) ;
	res = res || p == 1 ;

	// check if delta-minus contains pending changes
	GrB_OK (GrB_Matrix_get_INT32 (DM, &p, GxB_WILL_WAIT)) ;
	res = res || p == 1 ;

	// set output
	*willWait = res ;
	return GrB_SUCCESS ;
}

