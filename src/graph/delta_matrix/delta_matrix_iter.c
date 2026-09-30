/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "./delta_matrix_iter.h"
#include "./delta_matrix_iter_internal.h"
#include "../../util/rmalloc.h"
#include "./delta_utils.h"

// seek iterator within the given range of rows
// sets depleted to true if no values exist in the given range
static inline void _Iter_seek
(
	GxB_Iterator it,
	GrB_Index min_row,
	GrB_Index max_row,
	bool *depleted
) {
	GrB_Info info = GxB_rowIterator_seekRow (it, min_row) ;

	switch (info)
	{
		case GxB_EXHAUSTED:
			// no values to iterate on
			*depleted = true ;
			break ;

		case GrB_NO_VALUE:
			// in sparse matrix no value in the current row
			// seek to first none empty row
			while (info == GrB_NO_VALUE &&
				   GxB_rowIterator_getRowIndex (it) < max_row) {
				info = GxB_rowIterator_nextRow (it) ;
			}

			*depleted = (info != GrB_SUCCESS ||
						GxB_rowIterator_getRowIndex (it) > max_row) ;
			break ;

		case GrB_SUCCESS:
			// in hypersparse matrix iterator move to the next row with values
			// make sure seekRow didn't over-reached
			*depleted = GxB_rowIterator_getRowIndex (it) > max_row;
			break ;

		default:
			ASSERT (false) ;
			break ;
	}
}

// allocate, attach, and seek iterator
static inline void _Iter_attach
(
	GxB_Iterator it,    // iterator
	GrB_Matrix m,       // matrix to attach to
	GrB_Index min_row,  // starting row (inclusive)
	GrB_Index max_row,  // ending row (exclusive)
	bool *depleted      // true if no values in range
) {
	ASSERT (m        != NULL) ;
	ASSERT (it       != NULL) ;
	ASSERT (depleted != NULL) ;
	ASSERT (min_row  <= max_row) ;

	*depleted = true ; // default

	GrB_OK (GxB_rowIterator_attach (it, m, NULL)) ;
	_Iter_seek (it, min_row, max_row, depleted) ;
}

// advance internal iterator
void _Iter_next
(
	GxB_Iterator it,
	GrB_Index max_row,
	bool *depleted
) {
	GrB_Info   info ;

	info = GxB_rowIterator_nextCol (it) ;
	if (info != GrB_SUCCESS) {
		info = GxB_rowIterator_nextRow (it) ;
		// in-case iterator maintains number of yield values, we can use nvals here
		// for a quick return!
		while (info == GrB_NO_VALUE &&
			   GxB_rowIterator_getRowIndex (it) < max_row) {
			info = GxB_rowIterator_nextRow (it) ;
		}

		// prep for next call to `_next_m_iter`
		*depleted = info != GrB_SUCCESS ||
					GxB_rowIterator_getRowIndex(it) > max_row ;
	}
}

//------------------------------------------------------------------------------
// Delta_MaskedIter - M's row iterator, masked by pending deletions (DM)
//------------------------------------------------------------------------------

// advance mi past any M entry masked out by a pending deletion, so that
// whenever mi->depleted is false, mi->it's current position is a live
// entry, ready to be read without further filtering. Idempotent - a no-op
// if the invariant already holds.
static void _MaskedIter_skip_masked
(
	Delta_MaskedIter *mi,
	GrB_Index max_row
) {
	while (!mi->depleted && !mi->dm_depleted) {
		GrB_Index m_row = GxB_rowIterator_getRowIndex (&mi->it) ;
		GrB_Index d_row = GxB_rowIterator_getRowIndex (&mi->dm_it) ;

		if (m_row != d_row) {
			return ;
		}

		GrB_Index m_col = GxB_rowIterator_getColIndex (&mi->it) ;
		GrB_Index d_col = GxB_rowIterator_getColIndex (&mi->dm_it) ;

		if (m_col != d_col) {
			return ;
		}

		// current M entry is masked out by a pending deletion, skip both
		_Iter_next (&mi->it,    max_row, &mi->depleted) ;
		_Iter_next (&mi->dm_it, max_row, &mi->dm_depleted) ;
	}
}

// attach mi to M/DM and position at the first live entry in [min_row, max_row]
static void _MaskedIter_attach
(
	Delta_MaskedIter *mi,
	GrB_Matrix M,
	GrB_Matrix DM,
	GrB_Index min_row,
	GrB_Index max_row
) {
	_Iter_attach (&mi->it,    M,  min_row, max_row, &mi->depleted) ;
	_Iter_attach (&mi->dm_it, DM, min_row, max_row, &mi->dm_depleted) ;
	_MaskedIter_skip_masked (mi, max_row) ;
}

// reseek an already-attached mi to [min_row, max_row], without re-deriving
// M/DM's sparsity/format, and position at the first live entry
static void _MaskedIter_seek
(
	Delta_MaskedIter *mi,
	GrB_Index min_row,
	GrB_Index max_row
) {
	_Iter_seek (&mi->it,    min_row, max_row, &mi->depleted) ;
	_Iter_seek (&mi->dm_it, min_row, max_row, &mi->dm_depleted) ;
	_MaskedIter_skip_masked (mi, max_row) ;
}

// advance mi to its next live entry (or depleted)
void _MaskedIter_next
(
	Delta_MaskedIter *mi,
	GrB_Index max_row
) {
	_Iter_next (&mi->it, max_row, &mi->depleted) ;
	_MaskedIter_skip_masked (mi, max_row) ;
}

//------------------------------------------------------------------------------

// iterate over a single row
GrB_Info Delta_MatrixTupleIter_iterate_row
(
	Delta_MatrixTupleIter *iter,  //must be attached
	GrB_Index rowIdx              // row index to iterate
) {
	if (IS_DETACHED (iter)) {
		return GrB_NULL_POINTER ;
	}

	iter->min_row = rowIdx ;
	iter->max_row = rowIdx ;

	_MaskedIter_seek (&iter->m_it, rowIdx, rowIdx) ;
	_Iter_seek (&iter->dp_it, rowIdx, rowIdx, &iter->dp_depleted) ;

	return GrB_SUCCESS ;
}

// iterate over a range of rows
GrB_Info Delta_MatrixTupleIter_iterate_range
(
	Delta_MatrixTupleIter *iter,  // iterator to use
	GrB_Index startRowIdx,        // row index to start with (inclusive)
	GrB_Index endRowIdx           // row index to finish with (exclusive)
) {
	ASSERT (startRowIdx <= endRowIdx) ;

	if (IS_DETACHED (iter)) {
		return GrB_NULL_POINTER ;
	}

	iter->min_row = startRowIdx ;
	iter->max_row = endRowIdx ;

	_MaskedIter_seek (&iter->m_it, iter->min_row, iter->max_row) ;
	_Iter_seek (&iter->dp_it, iter->min_row, iter->max_row, &iter->dp_depleted) ;

	return GrB_SUCCESS ;
}

// generate a "next" function for the M matrix iterator
// T     - C type of the value,     e.g. uint64_t
// GB_T  - GraphBLAS type suffix,   e.g. UINT64
#define NEXT_M_ITER(T, GB_T)                                              \
static inline GrB_Info _next_m_iter_##GB_T                                \
(                                                                         \
	Delta_MatrixTupleIter *iter,  /* iterator scanning M             */   \
	GrB_Index *row,               /* optional extracted row index    */   \
	GrB_Index *col,               /* optional extracted column index */   \
	T *val                        /* optional extracted value        */   \
) {                                                                       \
	ASSERT (iter != NULL) ;                                               \
                                                                          \
	if (iter->m_it.depleted) {                                            \
		return GrB_NO_VALUE ;                                             \
	}                                                                     \
                                                                          \
	if (row) {                                                            \
		*row = GxB_rowIterator_getRowIndex (&iter->m_it.it) ;             \
	}                                                                     \
                                                                          \
	if (col) {                                                            \
		*col = GxB_rowIterator_getColIndex (&iter->m_it.it) ;             \
	}                                                                     \
                                                                          \
	if (val) {                                                            \
		*val = GxB_Iterator_get_##GB_T (&iter->m_it.it) ;                 \
	}                                                                     \
                                                                          \
	/* prep for next call */                                              \
	_MaskedIter_next (&iter->m_it, iter->max_row) ;                       \
                                                                          \
	return GrB_SUCCESS ;                                                  \
}

// static inline GrB_Info _next_m_iter_BOOL(iter, row, col, val)
NEXT_M_ITER(bool, BOOL)

// static inline GrB_Info _next_m_iter_UINT64(iter, row, col, val)
NEXT_M_ITER(uint64_t, UINT64)

#define NEXT_ITET(T, GB_T)                                                \
GrB_Info Delta_MatrixTupleIter_next_##GB_T                                \
(                                                                         \
	Delta_MatrixTupleIter *iter,  /* iterator to consume           */     \
	GrB_Index *row,               /* optional output row index     */     \
	GrB_Index *col,               /* optional output column index  */     \
	T *val                        /* optional value at A[row, col] */     \
) {                                                                       \
	if (IS_DETACHED (iter)) {                                             \
		return GrB_NULL_POINTER ;                                         \
	}                                                                     \
                                                                          \
	GrB_Info     info  =  GrB_SUCCESS  ;                                  \
	GxB_Iterator dp_it =  &iter->dp_it ;                                  \
                                                                          \
	if (!iter->m_it.depleted) {                                           \
		info = _next_m_iter_##GB_T (iter, row, col, val) ;                \
		if (info == GrB_SUCCESS) {                                        \
			return GrB_SUCCESS ;                                          \
		}                                                                 \
	}                                                                     \
                                                                          \
	if (iter->dp_depleted) {                                              \
		return GxB_EXHAUSTED ;                                            \
	}                                                                     \
                                                                          \
	if (row) {                                                            \
		*row = GxB_rowIterator_getRowIndex (dp_it) ;                      \
	}                                                                     \
                                                                          \
	if (col) {                                                            \
		*col = GxB_rowIterator_getColIndex (dp_it) ;                      \
	}                                                                     \
                                                                          \
	if (val) {                                                            \
		*val = GxB_Iterator_get_##GB_T (dp_it) ;                          \
	}                                                                     \
                                                                          \
	/* prep value for next iteration */                                   \
	_Iter_next (dp_it, iter->max_row, &iter->dp_depleted) ;               \
                                                                          \
	return GrB_SUCCESS ;                                                  \
}

// advance iterator
// GrB_Info Delta_MatrixTupleIter_next_BOOL (iter, row, col, val)
NEXT_ITET(bool, BOOL)

// GrB_Info Delta_MatrixTupleIter_next_UINT64 (iter, row, col, val)
NEXT_ITET(uint64_t, UINT64)

// reset iterator, assumes the iterator is valid
GrB_Info Delta_MatrixTupleIter_reset
(
	Delta_MatrixTupleIter *iter  // iterator to reset
) {
	if (IS_DETACHED (iter)) {
		return GrB_NULL_POINTER ;
	}

	_MaskedIter_seek (&iter->m_it, iter->min_row, iter->max_row) ;
	_Iter_seek (&iter->dp_it, iter->min_row, iter->max_row, &iter->dp_depleted) ;

	return GrB_SUCCESS ;
}

// returns true if iterator is attached to given matrix false otherwise
bool Delta_MatrixTupleIter_is_attached
(
	const Delta_MatrixTupleIter *iter,  // iterator to check
	const Delta_Matrix M                // matrix attached to
) {
	ASSERT (iter != NULL) ;

	return iter->A == M ;
}

// update iterator to scan given matrix
GrB_Info Delta_MatrixTupleIter_attach
(
	Delta_MatrixTupleIter *iter,  // iterator to update
	const Delta_Matrix A          // matrix to scan
) {
	return Delta_MatrixTupleIter_AttachRange (iter, A, DELTA_ITER_MIN_ROW,
		DELTA_ITER_MAX_ROW) ;
}

// update iterator to scan given matrix
GrB_Info Delta_MatrixTupleIter_AttachRange
(
	Delta_MatrixTupleIter *iter,  // iterator to update
	const Delta_Matrix A,         // matrix to scan
	GrB_Index min_row,            // minimum row for iteration
	GrB_Index max_row             // maximum row for iteration
) {
	if (A == NULL) {
		return GrB_NULL_POINTER ;
	}

	if (iter == NULL) {
		return GrB_NULL_POINTER ;
	}

	ASSERT (min_row <= max_row) ;

	GrB_Matrix M  = DELTA_MATRIX_M (A) ;
	GrB_Matrix DP = DELTA_MATRIX_DELTA_PLUS (A) ;
	GrB_Matrix DM = DELTA_MATRIX_DELTA_MINUS (A) ;

	iter->A = A ;
	iter->min_row = min_row ;
	iter->max_row = max_row ;

	_MaskedIter_attach (&iter->m_it, M, DM, iter->min_row, iter->max_row) ;
	_Iter_attach (&iter->dp_it, DP, iter->min_row, iter->max_row, &iter->dp_depleted) ;

	return GrB_SUCCESS ;
}

// return the position of the iterator
// p is read before next() advances the iterator, giving the position of
// the entry about to be returned; unique and stable across re-scans of
// the same matrix
//
// safe with respect to pending deletions: iter->m_it (a Delta_MaskedIter)
// guarantees that whenever iter->m_it.depleted is false, its current position
// is already a live (non-deleted) entry - see _MaskedIter_skip_masked,
// which every attach/seek/next on iter->m_it runs before returning - so this
// never reports the position of an entry masked out by DM.
//
// NOTE: the value returned is not necessarily increasing, nor are all values
// in a range returned (some p values between 0 and pmax may be skipped) the
// only guarantee is that p is unique to the current entry in the matrix, and
// that the same p value will be returned if a second iterator queries getp at
// the same position AND the current matrix has not changed
GrB_Index Delta_Matrix_Iterator_getp
(
	Delta_MatrixTupleIter *iter  // iterator to query
) {
	if (!iter->m_it.depleted) {
		return GxB_Matrix_Iterator_getp (&iter->m_it.it) ;
	}
	// M is exhausted: use nvals(M) as a fixed offset so dp positions
	// don't collide with any M entry position (0..nvals(M)-1)
	return GxB_Matrix_Iterator_getpmax (&iter->m_it.it)
		+ GxB_Matrix_Iterator_getp (&iter->dp_it) ;
}

// free iterator data
GrB_Info Delta_MatrixTupleIter_detach
(
	Delta_MatrixTupleIter *iter  // iterator to free
) {
	ASSERT(iter != NULL) ;

	iter->A                = NULL ;
	iter->dp_depleted      = true ;
	iter->m_it.depleted    = true ;
	iter->m_it.dm_depleted = true ;

	return GrB_SUCCESS ;
}

