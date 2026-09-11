/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "./delta_matrix_iter.h"
#include "../../util/rmalloc.h"
#include "./delta_utils.h"

// returns true if iterator is detached from a matrix
#define IS_DETACHED(iter) ((iter) == NULL || (iter)->A == NULL)

// seek iterator within the given range of rows
// sets depleted to true if no values exist in the given range
static inline void _set_iter_range
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
			while (info == GrB_NO_VALUE && GxB_rowIterator_getRowIndex (it) < max_row) {
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
static inline void _init_iter
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
	_set_iter_range (it, min_row, max_row, depleted) ;
}

// advance internal iterator
static void _iter_next
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
		while(info == GrB_NO_VALUE && GxB_rowIterator_getRowIndex(it) < max_row) {
			info = GxB_rowIterator_nextRow (it) ;
		}

		// prep for next call to `_next_m_iter`
		*depleted = info != GrB_SUCCESS || GxB_rowIterator_getRowIndex(it) > max_row ;
	}
}

//------------------------------------------------------------------------------
// Delta_MaskedMIter - M's row iterator, masked by pending deletions (DM)
//------------------------------------------------------------------------------

// advance mi past any M entry masked out by a pending deletion, so that
// whenever mi->depleted is false, mi->it's current position is a live
// entry, ready to be read without further filtering. Idempotent - a no-op
// if the invariant already holds.
static void _MaskedMIter_skip_masked
(
	Delta_MaskedMIter *mi,
	GrB_Index max_row
) {
	while (!mi->depleted && !mi->dm_depleted) {
		GrB_Index m_row = GxB_rowIterator_getRowIndex (&mi->it) ;
		GrB_Index d_row = GxB_rowIterator_getRowIndex (&mi->dm_it) ;

		GrB_Index m_col = GxB_rowIterator_getColIndex (&mi->it) ;
		GrB_Index d_col = GxB_rowIterator_getColIndex (&mi->dm_it) ;

		if (m_row != d_row || m_col != d_col) {
			return ;
		}

		// current M entry is masked out by a pending deletion, skip both
		_iter_next (&mi->it,    max_row, &mi->depleted) ;
		_iter_next (&mi->dm_it, max_row, &mi->dm_depleted) ;
	}
}

// attach mi to M/DM and position at the first live entry in [min_row, max_row]
static void _MaskedMIter_attach
(
	Delta_MaskedMIter *mi,
	GrB_Matrix M,
	GrB_Matrix DM,
	GrB_Index min_row,
	GrB_Index max_row
) {
	_init_iter (&mi->it,    M,  min_row, max_row, &mi->depleted) ;
	_init_iter (&mi->dm_it, DM, min_row, max_row, &mi->dm_depleted) ;
	_MaskedMIter_skip_masked (mi, max_row) ;
}

// reseek an already-attached mi to [min_row, max_row], without re-deriving
// M/DM's sparsity/format, and position at the first live entry
static void _MaskedMIter_seek
(
	Delta_MaskedMIter *mi,
	GrB_Index min_row,
	GrB_Index max_row
) {
	_set_iter_range (&mi->it,    min_row, max_row, &mi->depleted) ;
	_set_iter_range (&mi->dm_it, min_row, max_row, &mi->dm_depleted) ;
	_MaskedMIter_skip_masked (mi, max_row) ;
}

// advance mi to its next live entry (or depleted)
static void _MaskedMIter_next
(
	Delta_MaskedMIter *mi,
	GrB_Index max_row
) {
	_iter_next (&mi->it, max_row, &mi->depleted) ;
	_MaskedMIter_skip_masked (mi, max_row) ;
}

//------------------------------------------------------------------------------

// iterate over a single row
GrB_Info Delta_MatrixTupleIter_iterate_row
(
	Delta_MatrixTupleIter *iter,  //must be attached
	GrB_Index rowIdx              // row index to iterate
) {
	if(IS_DETACHED(iter)) return GrB_NULL_POINTER ;

	iter->min_row = rowIdx ;
	iter->max_row = rowIdx ;

	_MaskedMIter_seek (&iter->m, iter->min_row, iter->max_row) ;
	_set_iter_range (&iter->dp_it, iter->min_row, iter->max_row, &iter->dp_depleted) ;

	return GrB_SUCCESS ;
}

// iterate over a range of rows
GrB_Info Delta_MatrixTupleIter_iterate_range
(
	Delta_MatrixTupleIter *iter,  // iterator to use
	GrB_Index startRowIdx,        // row index to start with (inclusive)
	GrB_Index endRowIdx           // row index to finish with (exclusive)
) {
	if(IS_DETACHED(iter)) return GrB_NULL_POINTER ;
	ASSERT(startRowIdx <= endRowIdx) ;

	iter->min_row = startRowIdx ;
	iter->max_row = endRowIdx ;

	_MaskedMIter_seek (&iter->m, iter->min_row, iter->max_row) ;
	_set_iter_range (&iter->dp_it, iter->min_row, iter->max_row, &iter->dp_depleted) ;

	return GrB_SUCCESS ;
}

// iterate over M matrix
static GrB_Info _next_m_iter_bool
(
	Delta_MatrixTupleIter *iter,  // iterator scanning M
	GrB_Index *row,               // optional extracted row index
	GrB_Index *col,               // optional extracted column index
	bool *val                     // optional extracted value
) {
	ASSERT(iter != NULL) ;

	if(iter->m.depleted) return GrB_NO_VALUE ;

	if(row) *row = GxB_rowIterator_getRowIndex (&iter->m.it) ;
	if(col) *col = GxB_rowIterator_getColIndex (&iter->m.it) ;
	if(val) *val = GxB_Iterator_get_BOOL (&iter->m.it) ;

	// prep for next call
	_MaskedMIter_next (&iter->m, iter->max_row) ;

	return GrB_SUCCESS ;
}

// advance iterator
GrB_Info Delta_MatrixTupleIter_next_BOOL
(
	Delta_MatrixTupleIter *iter,  // iterator to consume
	GrB_Index *row,               // optional output row index
	GrB_Index *col,               // optional output column index
	bool *val                     // optional value at A[row, col]
) {
	if(IS_DETACHED(iter)) return GrB_NULL_POINTER ;

	GrB_Info             info     =  GrB_SUCCESS  ;
	GxB_Iterator         dp_it    =  &iter->dp_it ;

	if(!iter->m.depleted) {
		info = _next_m_iter_bool(iter, row, col, val) ;
		if(info == GrB_SUCCESS) return GrB_SUCCESS ;
	}

	if(iter->dp_depleted) {
		return GxB_EXHAUSTED ;
	}

	if(row) *row = GxB_rowIterator_getRowIndex (dp_it) ;
	if(col) *col = GxB_rowIterator_getColIndex (dp_it) ;
	if(val) *val = GxB_Iterator_get_BOOL (dp_it) ;

	// prep value for next iteration
	_iter_next(dp_it, iter->max_row, &iter->dp_depleted);

	return GrB_SUCCESS ;
}

// iterate over M matrix
static GrB_Info _next_m_iter_uint64
(
	Delta_MatrixTupleIter *iter,  // iterator scanning M
	GrB_Index *row,               // optional extracted row index
	GrB_Index *col,               // optional extracted column index
	uint64_t *val                 // optional extracted value
) {
	ASSERT(iter != NULL) ;

	if(iter->m.depleted) return GrB_NO_VALUE ;

	if(row) *row = GxB_rowIterator_getRowIndex (&iter->m.it) ;
	if(col) *col = GxB_rowIterator_getColIndex (&iter->m.it) ;
	if(val) *val = GxB_Iterator_get_UINT64 (&iter->m.it) ;

	// prep for next call - already skips any masked entry
	_MaskedMIter_next (&iter->m, iter->max_row) ;

	return GrB_SUCCESS ;
}

// advance iterator
GrB_Info Delta_MatrixTupleIter_next_UINT64
(
	Delta_MatrixTupleIter *iter,  // iterator to consume
	GrB_Index *row,               // optional output row index
	GrB_Index *col,               // optional output column index
	uint64_t *val                 // optional value at A[row, col]
) {
	if(IS_DETACHED(iter)) return GrB_NULL_POINTER ;

	GrB_Info      info   =  GrB_SUCCESS                    ;
	GxB_Iterator  dp_it  =  &iter->dp_it                    ;

	if(!iter->m.depleted) {
		info = _next_m_iter_uint64(iter, row, col, val) ;
		if(info == GrB_SUCCESS) return GrB_SUCCESS ;
	}

	if(iter->dp_depleted) {
		return GxB_EXHAUSTED ;
	}

	if(row) *row = GxB_rowIterator_getRowIndex (dp_it) ;
	if(col) *col = GxB_rowIterator_getColIndex (dp_it) ;
	if(val) *val = GxB_Iterator_get_UINT64 (dp_it) ;

	// prep value for next iteration
	_iter_next(dp_it, iter->max_row, &iter->dp_depleted);

	return GrB_SUCCESS ;
}

// advance iterator in true ascending (row, col) order across the whole
// attached range.
//
// Delta_MatrixTupleIter_next_BOOL/_UINT64 drain the entire main matrix (M)
// before ever yielding a delta-plus (DP) entry, so within one attached
// range the stream is "all of M ascending, then all of DP ascending" - not
// one merged ascending stream. A caller that resumes a range scan (e.g.
// batch-based index population, re-attaching at [last_row + 1, MAX) between
// batches) can silently skip DP rows below the resume point forever, since
// they're outside every subsequent range.
#define DELTA_MATRIX_TUPLE_ITER_NEXT_SORTED(SUFFIX, TYPE)                      \
GrB_Info Delta_MatrixTupleIter_next_##SUFFIX##_sorted                          \
(                                                                              \
	Delta_MatrixTupleIter *iter,  /* iterator to consume */                    \
	GrB_Index *row,               /* optional output row index */              \
	GrB_Index *col,               /* optional output column index */           \
	TYPE *val                     /* optional value at A[row, col] */          \
) {                                                                            \
	if(IS_DETACHED(iter)) return GrB_NULL_POINTER ;                            \
                                                                               \
	bool have_m  = !iter->m.depleted ;                                         \
	bool have_dp = !iter->dp_depleted ;                                        \
                                                                               \
	if(!have_m && !have_dp) {                                                  \
		return GxB_EXHAUSTED ;                                                 \
	}                                                                          \
                                                                               \
	GxB_Iterator m_it  = &iter->m.it ;                                         \
	GxB_Iterator dp_it = &iter->dp_it ;                                        \
                                                                               \
	GrB_Index m_row = 0, m_col = 0, dp_row = 0, dp_col = 0 ;                   \
	if(have_m)  {                                                              \
		m_row  = GxB_rowIterator_getRowIndex (m_it) ;                          \
		m_col  = GxB_rowIterator_getColIndex (m_it) ;                          \
	}                                                                          \
	if(have_dp)  {                                                             \
		dp_row  = GxB_rowIterator_getRowIndex (dp_it) ;                        \
		dp_col  = GxB_rowIterator_getColIndex (dp_it) ;                        \
	}                                                                          \
                                                                               \
	/* take from DP only if it strictly precedes M's current entry */          \
	bool take_dp = have_dp &&                                                  \
		(!have_m || dp_row < m_row || (dp_row == m_row && dp_col < m_col)) ;   \
                                                                               \
	if(take_dp) {                                                              \
		if(row) *row = dp_row ;                                                \
		if(col) *col = dp_col ;                                                \
		if(val) *val = GxB_Iterator_get_##SUFFIX (dp_it) ;                     \
		_iter_next (dp_it, iter->max_row, &iter->dp_depleted) ;                \
	} else {                                                                   \
		if(row) *row = m_row ;                                                 \
		if(col) *col = m_col ;                                                 \
		if(val) *val = GxB_Iterator_get_##SUFFIX (m_it) ;                      \
		_MaskedMIter_next (&iter->m, iter->max_row) ;                          \
	}                                                                          \
                                                                               \
	return GrB_SUCCESS ;                                                       \
}

DELTA_MATRIX_TUPLE_ITER_NEXT_SORTED(BOOL, bool)
DELTA_MATRIX_TUPLE_ITER_NEXT_SORTED(UINT64, uint64_t)

#undef DELTA_MATRIX_TUPLE_ITER_NEXT_SORTED

// reset iterator, assumes the iterator is valid
GrB_Info Delta_MatrixTupleIter_reset
(
	Delta_MatrixTupleIter *iter  // iterator to reset
) {
	if(IS_DETACHED(iter)) return GrB_NULL_POINTER ;

	_MaskedMIter_seek (&iter->m, iter->min_row, iter->max_row) ;
	_set_iter_range (&iter->dp_it, iter->min_row, iter->max_row, &iter->dp_depleted) ;

	return GrB_SUCCESS ;
}

// returns true if iterator is attached to given matrix false otherwise
bool Delta_MatrixTupleIter_is_attached
(
	const Delta_MatrixTupleIter *iter,  // iterator to check
	const Delta_Matrix M                // matrix attached to
) {
	ASSERT(iter != NULL);

	return iter->A == M;
}

// update iterator to scan given matrix
GrB_Info Delta_MatrixTupleIter_attach
(
	Delta_MatrixTupleIter *iter,  // iterator to update
	const Delta_Matrix A          // matrix to scan
) {
	return Delta_MatrixTupleIter_AttachRange(iter, A, DELTA_ITER_MIN_ROW,
		DELTA_ITER_MAX_ROW);
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

	_MaskedMIter_attach (&iter->m, M, DM, iter->min_row, iter->max_row) ;
	_init_iter (&iter->dp_it, DP, iter->min_row, iter->max_row, &iter->dp_depleted) ;

	return GrB_SUCCESS ;
}

// return the position of the iterator
// p is read before next() advances the iterator, giving the position of
// the entry about to be returned; unique and stable across re-scans of
// the same matrix
GrB_Index Delta_Matrix_Iterator_getp
(
	Delta_MatrixTupleIter *iter  // iterator to query
) {
	if (!iter->m.depleted) {
		return GxB_Matrix_Iterator_getp (&iter->m.it) ;
	}
	// M is exhausted: use nvals(M) as a fixed offset so dp positions
	// don't collide with any M entry position (0..nvals(M)-1)
	return GxB_Matrix_Iterator_getpmax (&iter->m.it)
		+ GxB_Matrix_Iterator_getp (&iter->dp_it) ;
}

// free iterator data
GrB_Info Delta_MatrixTupleIter_detach
(
	Delta_MatrixTupleIter *iter  // iterator to free
) {
	ASSERT(iter != NULL) ;

	iter->A            = NULL ;
	iter->m.depleted   = true ;
	iter->dp_depleted  = true ;
	iter->m.dm_depleted = true ;

	return GrB_SUCCESS ;
}
