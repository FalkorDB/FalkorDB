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
	GrB_RETURN_IF_FAIL (GrB_Matrix_nvals (&nvals, dp)) ;

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

	GrB_Matrix M  = DELTA_MATRIX_M (C) ;
	GrB_Matrix DP = DELTA_MATRIX_DELTA_PLUS (C) ;
	GrB_Matrix DM = DELTA_MATRIX_DELTA_MINUS (C) ;

	if (force_sync) {
		GrB_RETURN_IF_FAIL (Delta_Matrix_sync_deletions (C)) ;
		GrB_RETURN_IF_FAIL (Delta_Matrix_sync_additions (C)) ;
	} else {
		GrB_Index dp_nvals = 0;
		GrB_Index dm_nvals = 0;

		//----------------------------------------------------------------------
		// determine change set
		//----------------------------------------------------------------------

		GrB_RETURN_IF_FAIL (GrB_Matrix_nvals (&dp_nvals, DP)) ;
		GrB_RETURN_IF_FAIL (GrB_Matrix_nvals (&dm_nvals, DM)) ;

		//----------------------------------------------------------------------
		// flush deletions
		//----------------------------------------------------------------------

		if (dm_nvals >= delta_max_pending_changes) {
			GrB_RETURN_IF_FAIL (Delta_Matrix_sync_deletions (C)) ;
		}

		//----------------------------------------------------------------------
		// flush additions
		//----------------------------------------------------------------------

		if (dp_nvals >= delta_max_pending_changes) {
			GrB_RETURN_IF_FAIL (Delta_Matrix_sync_additions (C)) ;
		}
	}

	// wait on all 3 matrices
	GrB_RETURN_IF_FAIL (GrB_wait (M,  GrB_MATERIALIZE)) ;
	GrB_RETURN_IF_FAIL (GrB_wait (DM, GrB_MATERIALIZE)) ;
	GrB_RETURN_IF_FAIL (GrB_wait (DP, GrB_MATERIALIZE)) ;

	// C shouldn't have any pending operations
	ASSERT (!Delta_Matrix_willWait (C)) ;

	return GrB_SUCCESS ;
}

GrB_Info Delta_Matrix_wait
(
	Delta_Matrix A,
	bool force_sync
) {
	ASSERT (A != NULL) ;

	if (DELTA_MATRIX_MAINTAIN_TRANSPOSE (A)) {
		GrB_RETURN_IF_FAIL (Delta_Matrix_wait (A->transposed, force_sync)) ;
	}

	uint64_t delta_max_pending_changes ;
	Config_Option_get (Config_DELTA_MAX_PENDING_CHANGES,
			&delta_max_pending_changes) ;

	GrB_RETURN_IF_FAIL (Delta_Matrix_sync (A, force_sync,
				delta_max_pending_changes)) ;

	return GrB_SUCCESS ;
}

// synchronizes the DeltaMatrix `C`
// in case `C` isn't of the expected dimensions it will be resized
// in case GraphBLAS indicates one of `C`'s internal matrices: `M`, `DP` or `DM`
// requires waiting then these matrices will be synchronized
GrB_Info Delta_Matrix_synchronize
(
	Delta_Matrix C,   // the DeltaMatrix to synchronize
	GrB_Index nrows,  // the required number of rows
	GrB_Index ncols   // the required number of columns
)
{
	ASSERT (C != NULL) ;

	GrB_Info info = GrB_SUCCESS ;
	uint64_t C_nrows = 0 ;
	uint64_t C_ncols = 0 ;

	Delta_Matrix_lock (C) ;

	//--------------------------------------------------------------------------
	// get C's number of rows and columns
	//--------------------------------------------------------------------------

	info = Delta_Matrix_nrows (&C_nrows, C) ;
	if (info != GrB_SUCCESS) {
		goto unlock ;
	}

	info = Delta_Matrix_ncols (&C_ncols, C) ;
	if (info != GrB_SUCCESS) {
		goto unlock ;
	}

	bool will_wait    = Delta_Matrix_willWait (C) ;
	bool already_sync = (C_nrows >= nrows &&
						 C_ncols >= ncols &&
						 !will_wait) ;

	if (already_sync == true) {
		goto unlock ;
	}

	if (C_nrows < nrows || C_ncols < ncols) {
		info = Delta_Matrix_resize (C, nrows, ncols) ;
		if (info != GrB_SUCCESS) {
			// failed to resize
			goto unlock ;
		}
	}

	if (will_wait) {
		info = Delta_Matrix_wait (C, false) ;
		ASSERT ((info == GrB_SUCCESS && Delta_Matrix_willWait (C) == false) ||
				 info != GrB_SUCCESS) ;
	}

unlock:
	Delta_Matrix_unlock (C) ;

	return info ;
}

