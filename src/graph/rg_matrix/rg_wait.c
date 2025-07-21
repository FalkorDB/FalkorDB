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
#include "../../util/rmalloc.h"
#include "configuration/config.h"

static inline void _SetUndirty
(
	RG_Matrix C
) {
	ASSERT(C);

	C->dirty = false;

	if(RG_MATRIX_MAINTAIN_TRANSPOSE(C)) {
		C->transposed->dirty = false;
	}
}

static void RG_Matrix_sync_deletions
(
	RG_Matrix C
) {
	ASSERT(C != NULL);

	GrB_Matrix m  = RG_MATRIX_M(C);
	GrB_Matrix dm = RG_MATRIX_DELTA_MINUS(C);

	GrB_Info info;

	info = GrB_transpose(m, dm, GrB_NULL, m, GrB_DESC_RSCT0);
	ASSERT(info == GrB_SUCCESS);

	// clear delta minus
	info = GrB_Matrix_clear(dm);
	ASSERT(info == GrB_SUCCESS);
}

static void RG_Matrix_sync_additions
(
	RG_Matrix C
) {
	ASSERT(C != NULL);

	GrB_Matrix m  = RG_MATRIX_M(C);
	GrB_Matrix dp = RG_MATRIX_DELTA_PLUS(C);

	GrB_Info info;
	GrB_Index nrows;
	GrB_Index ncols;

	info = GrB_Matrix_nrows(&nrows, m);
	ASSERT(info == GrB_SUCCESS);
	info = GrB_Matrix_ncols(&ncols, m);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_assign(m, dp, NULL, dp, GrB_ALL, nrows, GrB_ALL, ncols,
		GrB_DESC_S);
	ASSERT(info == GrB_SUCCESS);

	// clear delta plus
	info = GrB_Matrix_clear(dp);
	ASSERT(info == GrB_SUCCESS);
}

static void RG_Matrix_sync
(
	RG_Matrix C,
	bool force_sync,
	uint64_t delta_max_pending_changes
) {
	ASSERT(C != NULL);

	GrB_Matrix m  = RG_MATRIX_M(C);
	GrB_Matrix dp = RG_MATRIX_DELTA_PLUS(C);
	GrB_Matrix dm = RG_MATRIX_DELTA_MINUS(C);

	if(force_sync) {
		RG_Matrix_sync_deletions(C);
		RG_Matrix_sync_additions(C);
	} else {
		GrB_Index dp_nvals;
		GrB_Index dm_nvals;

		//----------------------------------------------------------------------
		// determin change set
		//----------------------------------------------------------------------

		GrB_Matrix_nvals(&dp_nvals, dp);
		GrB_Matrix_nvals(&dm_nvals, dm);

		//----------------------------------------------------------------------
		// perform deletions
		//----------------------------------------------------------------------

		if(dm_nvals >= delta_max_pending_changes) {
			RG_Matrix_sync_deletions(C);
		}

		//----------------------------------------------------------------------
		// perform additions
		//----------------------------------------------------------------------

		if(dp_nvals >= delta_max_pending_changes) {
			RG_Matrix_sync_additions(C);
		}
	}

	// wait on all 3 matrices
	GrB_Info info = GrB_wait(m, GrB_MATERIALIZE);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_wait(dm, GrB_MATERIALIZE);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_wait(dp, GrB_MATERIALIZE);
	ASSERT(info == GrB_SUCCESS);
}

GrB_Info RG_Matrix_wait
(
	RG_Matrix A,
	bool force_sync
) {
	ASSERT(A != NULL);
	if(RG_MATRIX_MAINTAIN_TRANSPOSE(A)) {
		RG_Matrix_wait(A->transposed, force_sync);
	}

	uint64_t delta_max_pending_changes;
	Config_Option_get(Config_DELTA_MAX_PENDING_CHANGES,
			&delta_max_pending_changes);

	RG_Matrix_sync(A, force_sync, delta_max_pending_changes);

	_SetUndirty(A);

	return GrB_SUCCESS;
}

