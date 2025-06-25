/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_utils.h"
#include "delta_matrix.h"
#include "../../util/rmalloc.h"
#include "configuration/config.h"

static inline void _SetUndirty
(
	Delta_Matrix C
) {
	ASSERT(C);

	C->dirty = false;

	if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(C)) {
		C->transposed->dirty = false;
	}
}

static void Delta_Matrix_sync_deletions
(
	Delta_Matrix C
) {
	ASSERT(C != NULL);

	GrB_Matrix m  = DELTA_MATRIX_M(C);
	GrB_Matrix dm = DELTA_MATRIX_DELTA_MINUS(C);

	GrB_Info info;

	info = GrB_transpose(m, dm, GrB_NULL, m, GrB_DESC_RSCT0);
	ASSERT(info == GrB_SUCCESS);

	// clear delta minus
	info = GrB_Matrix_clear(dm);
	ASSERT(info == GrB_SUCCESS);
}

static void Delta_Matrix_sync_additions
(
	Delta_Matrix C
) {
	ASSERT(C != NULL);

	GrB_Matrix m  = DELTA_MATRIX_M(C);
	GrB_Matrix dp = DELTA_MATRIX_DELTA_PLUS(C);

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

static void Delta_Matrix_sync
(
	Delta_Matrix C,
	bool force_sync,
	uint64_t delta_max_pending_changes
) {
	ASSERT(C != NULL);

	GrB_Matrix m  = DELTA_MATRIX_M(C);
	GrB_Matrix dp = DELTA_MATRIX_DELTA_PLUS(C);
	GrB_Matrix dm = DELTA_MATRIX_DELTA_MINUS(C);

	if(force_sync) {
		Delta_Matrix_sync_deletions(C);
		Delta_Matrix_sync_additions(C);
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
			Delta_Matrix_sync_deletions(C);
		}

		//----------------------------------------------------------------------
		// perform additions
		//----------------------------------------------------------------------

		if(dp_nvals >= delta_max_pending_changes) {
			Delta_Matrix_sync_additions(C);
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

GrB_Info Delta_Matrix_wait
(
	Delta_Matrix A,
	bool force_sync
) {
	ASSERT(A != NULL);
	Delta_Matrix_validate(A);
	if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(A)) {
		Delta_Matrix_wait(A->transposed, force_sync);
	}

	uint64_t delta_max_pending_changes;
	Config_Option_get(Config_DELTA_MAX_PENDING_CHANGES,
			&delta_max_pending_changes);

	Delta_Matrix_sync(A, force_sync, delta_max_pending_changes);

	_SetUndirty(A);

	Delta_Matrix_validate(A);
	return GrB_SUCCESS;
}

void Delta_Matrix_synchronize
(
	Delta_Matrix A,
	GrB_Index nrows,
	GrB_Index ncols
)
{
	ASSERT(A != NULL);
	Delta_Matrix_validate(A);
	uint64_t A_nrows = 0;
	uint64_t A_ncols = 0;
	GrB_Info info = Delta_Matrix_nrows(&A_nrows, A);
	ASSERT(info == GrB_SUCCESS);
	info = Delta_Matrix_ncols(&A_ncols, A);
	ASSERT(info == GrB_SUCCESS);
	
	if (!(A_nrows < nrows || A_ncols < ncols || A->dirty)) {
		return;
	}
	Delta_Matrix_lock(A);
	info = Delta_Matrix_nrows(&A_nrows, A);
	ASSERT(info == GrB_SUCCESS);
	info = Delta_Matrix_ncols(&A_ncols, A);
	ASSERT(info == GrB_SUCCESS);
	if(A_nrows < nrows || A_ncols < ncols) {
		info = Delta_Matrix_resize(A, nrows, ncols);
		ASSERT(info == GrB_SUCCESS);
	}
	if(A->dirty) {
		info = Delta_Matrix_wait(A, false);
		ASSERT(info == GrB_SUCCESS);
	}
	Delta_Matrix_unlock(A);
	Delta_Matrix_validate(A);
}

