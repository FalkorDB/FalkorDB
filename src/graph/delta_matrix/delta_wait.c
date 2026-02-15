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

static GrB_Info Delta_Matrix_sync_deletions
(
	Delta_Matrix C
) {
	ASSERT (C != NULL) ;

	GrB_Matrix m  = DELTA_MATRIX_M (C) ;
	GrB_Matrix dm = DELTA_MATRIX_DELTA_MINUS (C) ;

	GrB_Index nvals ;
	GrB_RETURN_IF_FAIL (GrB_Matrix_nvals (&nvals, dm)) ;

	if (nvals > 0) { //shortcut if no vals
		// turn on burble and look for: "alias duplicate"
		// in that case replace with: asign
		// or assign an empty scalar with a struct mask
		GrB_RETURN_IF_FAIL (GrB_transpose (m, dm, NULL, m, GrB_DESC_RSCT0)) ;
	}

	// clear delta minus
	return GrB_Matrix_clear (dm) ;
}

static GrB_Info Delta_Matrix_sync_additions
(
	Delta_Matrix C
) {
	ASSERT (C != NULL) ;

	GrB_Matrix m  = DELTA_MATRIX_M (C) ;
	GrB_Matrix dp = DELTA_MATRIX_DELTA_PLUS (C) ;

	GrB_Index nvals ;
	GrB_RETURN_IF_FAIL (GrB_Matrix_nvals (&nvals, dp))

	if (nvals > 0) { //shortcut if no vals
		// TODO: turn on burble, see if "wait add pending tuples into existing A" shows up
		// in that case change to ewiseadd
		GrB_RETURN_IF_FAIL (GrB_Matrix_assign (m, dp, NULL, dp, GrB_ALL, 0,
					GrB_ALL, 0, GrB_DESC_S)) ;
	}

	// clear delta plus
	return GrB_Matrix_clear (dp) ;
}

static GrB_Info Delta_Matrix_sync
(
	Delta_Matrix C,
	bool force_sync,
	uint64_t delta_max_pending_changes
) {
	ASSERT (C != NULL) ;

	GrB_Matrix m  = DELTA_MATRIX_M (C) ;
	GrB_Matrix dp = DELTA_MATRIX_DELTA_PLUS (C) ;
	GrB_Matrix dm = DELTA_MATRIX_DELTA_MINUS (C) ;

	if (force_sync) {
		GrB_RETURN_IF_FAIL (Delta_Matrix_sync_deletions (C))
		GrB_RETURN_IF_FAIL (Delta_Matrix_sync_additions (C))
	} else {
		GrB_Index dp_nvals = 0;
		GrB_Index dm_nvals = 0;

		//----------------------------------------------------------------------
		// determine change set
		//----------------------------------------------------------------------

		GrB_RETURN_IF_FAIL (GrB_Matrix_nvals (&dp_nvals, dp))
		GrB_RETURN_IF_FAIL (GrB_Matrix_nvals (&dm_nvals, dm))

		//----------------------------------------------------------------------
		// perform deletions
		//----------------------------------------------------------------------

		if (dm_nvals >= delta_max_pending_changes) {
			GrB_RETURN_IF_FAIL (Delta_Matrix_sync_deletions (C))
		}

		//----------------------------------------------------------------------
		// perform additions
		//----------------------------------------------------------------------

		if (dp_nvals >= delta_max_pending_changes) {
			GrB_RETURN_IF_FAIL (Delta_Matrix_sync_additions (C))
		}
	}

	// wait on all 3 matrices
	GrB_RETURN_IF_FAIL (GrB_wait (m,  GrB_MATERIALIZE))
	GrB_RETURN_IF_FAIL (GrB_wait (dm, GrB_MATERIALIZE))
	GrB_RETURN_IF_FAIL (GrB_wait (dp, GrB_MATERIALIZE))

	return GrB_SUCCESS ;
}

GrB_Info Delta_Matrix_wait
(
	Delta_Matrix A,
	bool force_sync
) {
	ASSERT (A != NULL) ;

	if (DELTA_MATRIX_MAINTAIN_TRANSPOSE (A)) {
		GrB_RETURN_IF_FAIL (Delta_Matrix_wait (A->transposed, force_sync))
	}

	uint64_t delta_max_pending_changes ;
	Config_Option_get (Config_DELTA_MAX_PENDING_CHANGES,
			&delta_max_pending_changes) ;

	GrB_RETURN_IF_FAIL (Delta_Matrix_sync (A, force_sync,
				delta_max_pending_changes))

	_SetUndirty (A) ;

	return GrB_SUCCESS ;
}

GrB_Info Delta_Matrix_synchronize
(
	Delta_Matrix A,
	GrB_Index nrows,
	GrB_Index ncols
)
{
	ASSERT (A != NULL) ;
	uint64_t A_nrows = 0 ;
	uint64_t A_ncols = 0 ;

	GrB_RETURN_IF_FAIL (Delta_Matrix_nrows (&A_nrows, A)) ;
	GrB_RETURN_IF_FAIL (Delta_Matrix_ncols (&A_ncols, A)) ;
	
	bool alreay_sync = (A_nrows >= nrows && A_ncols >= ncols && !A->dirty) ;
	if (alreay_sync == true) {
		return GrB_SUCCESS ;
	}

	Delta_Matrix_lock (A) ;

	GrB_Info info ;

	// refetch under lock

	info = Delta_Matrix_nrows (&A_nrows, A) ;
	if (info != GrB_SUCCESS) {
		goto unlock ;
	}

	info = Delta_Matrix_ncols (&A_ncols, A) ;
	if (info != GrB_SUCCESS) {
		goto unlock ;
	}

	if (A_nrows < nrows || A_ncols < ncols) {
		info = Delta_Matrix_resize (A, nrows, ncols) ;
		if (info != GrB_SUCCESS) {
			goto unlock ;
		}
	}

	if (A->dirty) {
		info = Delta_Matrix_wait (A, false) ;
	}

unlock:
	Delta_Matrix_unlock (A) ;

	return info ;
}

