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
	int         p       =  false;
	bool        res     =  false;
	GrB_Matrix  M       =  DELTA_MATRIX_M(C);
	GrB_Matrix  DP      =  DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix  DM      =  DELTA_MATRIX_DELTA_MINUS(C);

	if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(C)) {
		info = Delta_Matrix_pending(C->transposed, &res);
		ASSERT(info == GrB_SUCCESS);
		if(res == true) {
			*pending = true;
			return GrB_SUCCESS;
		}
	}

	// check if M contains pending changes
	info = GrB_Matrix_get_INT32(M, &p, GxB_WILL_WAIT);
	ASSERT(info == GrB_SUCCESS);
	res = res || p == 1;

	// check if delta-plus contains pending changes
	info = GrB_Matrix_get_INT32(DP, &p, GxB_WILL_WAIT);
	ASSERT(info == GrB_SUCCESS);
	res = res || p == 1;

	// check if delta-plus contains pending changes
	info = GrB_Matrix_get_INT32(DM, &p, GxB_WILL_WAIT);
	ASSERT(info == GrB_SUCCESS);
	res = res || p == 1;

	// set output
	*pending = res;

	*pending = C->dirty;
	return info;
}

