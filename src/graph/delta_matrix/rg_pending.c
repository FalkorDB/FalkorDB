/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"

GrB_Info Delta_Matrix_pending
(
	const Delta_Matrix C,              // matrix to query
	bool *pending                   // are there any pending operations
) {
	ASSERT(C       != NULL);
	ASSERT(pending != NULL);

	GrB_Info    info;
	bool        p        =  false;
	bool        res      =  false;
	GrB_Matrix  M        =  DELTA_MATRIX_M(C);
	GrB_Matrix  DP       =  DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix  DM       =  DELTA_MATRIX_DELTA_MINUS(C);

	if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(C)) {
		info = Delta_Matrix_pending(C->transposed, &res);
		ASSERT(info == GrB_SUCCESS);
		if(res == true) {
			*pending = true;
			return GrB_SUCCESS;
		}
	}

	// // check if M contains pending changes
	// info = GxB_Matrix_Pending(M, &p);
	// ASSERT(info == GrB_SUCCESS);
	// res |= p;

	// // check if delta-plus contains pending changes
	// info = GxB_Matrix_Pending(DP, &p);
	// ASSERT(info == GrB_SUCCESS);
	// res |= p;

	// // check if delta-plus contains pending changes
	// info = GxB_Matrix_Pending(DM, &p);
	// ASSERT(info == GrB_SUCCESS);
	// res |= p;

	// // set output
	// *pending = res;

	*pending = C->dirty;
	return info;
}

