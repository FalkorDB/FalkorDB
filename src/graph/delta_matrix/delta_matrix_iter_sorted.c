/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "./delta_matrix_iter.h"
#include "./delta_matrix_iter_internal.h"

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
//
// This merges M (with pending deletions masked out via Delta_MaskedMIter,
// same as Delta_MatrixTupleIter_next_BOOL/_UINT64) and DP so that rows/cols
// only ever increase across calls - a caller resuming at [last_row + 1, MAX)
// is guaranteed nothing at or before last_row remains unvisited, from
// either M or DP.
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
		_Iter_next (dp_it, iter->max_row, &iter->dp_depleted) ;                \
	} else {                                                                   \
		if(row) *row = m_row ;                                                 \
		if(col) *col = m_col ;                                                 \
		if(val) *val = GxB_Iterator_get_##SUFFIX (m_it) ;                      \
		_MaskedMIter_next (&iter->m, iter->max_row) ;                          \
	}                                                                          \
                                                                               \
	return GrB_SUCCESS ;                                                       \
}

DELTA_MATRIX_TUPLE_ITER_NEXT_SORTED (BOOL, bool)
DELTA_MATRIX_TUPLE_ITER_NEXT_SORTED (UINT64, uint64_t)

#undef DELTA_MATRIX_TUPLE_ITER_NEXT_SORTED
