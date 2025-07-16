/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "rg_matrix.h"

GrB_Info RG_Matrix_pending
(
	const RG_Matrix C,              // matrix to query
	bool *pending                   // are there any pending operations
) {
	ASSERT(C       != NULL);
	ASSERT(pending != NULL);

	GrB_Info    info;
	bool        p        =  false;
	bool        res      =  false;
	GrB_Matrix  M        =  RG_MATRIX_M(C);
	GrB_Matrix  DP       =  RG_MATRIX_DELTA_PLUS(C);
	GrB_Matrix  DM       =  RG_MATRIX_DELTA_MINUS(C);

	if(RG_MATRIX_MAINTAIN_TRANSPOSE(C)) {
		info = RG_Matrix_pending(C->transposed, &res);
		ASSERT(info == GrB_SUCCESS);
		if(res == true) {
			*pending = true;
			return GrB_SUCCESS;
		}
	}

	// check if M contains pending changes
	info = GxB_Matrix_Pending(M, &p);
	ASSERT(info == GrB_SUCCESS);
	res |= p;

	// check if delta-plus contains pending changes
	info = GxB_Matrix_Pending(DP, &p);
	ASSERT(info == GrB_SUCCESS);
	res |= p;

	// check if delta-plus contains pending changes
	info = GxB_Matrix_Pending(DM, &p);
	ASSERT(info == GrB_SUCCESS);
	res |= p;

	// set output
	*pending = res;

	return info;
}

