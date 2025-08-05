/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "./delta_matrix_iter.h"
#include "../../util/rmalloc.h"

// returns true if iterator is detached from a matrix
#define IS_DETACHED(iter) ((iter) == NULL || (iter)->A == NULL)

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
			while (info == GrB_NO_VALUE && GxB_rowIterator_getRowIndex(it) < max_row) {
				info = GxB_rowIterator_nextRow (it) ;
			}

			*depleted = (info != GrB_SUCCESS ||
						GxB_rowIterator_getRowIndex(it) > max_row) ;
			break ;
		case GrB_SUCCESS:
			// in hypersparse matrix iterator move to the next row with values
			// make sure seekRow didn't over-reached
			*depleted = GxB_rowIterator_getRowIndex (it) > max_row;
			break ;		
		default:
			ASSERT(false);
			break;
	}
}

static inline void _init_iter
(
	GxB_Iterator it,
	GrB_Matrix m,
	GrB_Index min_row,
	GrB_Index max_row,
	bool *depleted
) {
	ASSERT(it       != NULL) ;
	ASSERT(m        != NULL) ;
	ASSERT(min_row  <= max_row) ;
	ASSERT(depleted != NULL) ;

	*depleted = true ; // default

	GrB_Info info ;
	UNUSED(info) ;

	info = GxB_rowIterator_attach(it, m, NULL) ;
	ASSERT(info == GrB_SUCCESS) ;
	_set_iter_range(it, min_row, max_row, depleted) ;
}

GrB_Info Delta_MatrixTupleIter_iterate_row
(
	Delta_MatrixTupleIter *iter,
	GrB_Index rowIdx
) {
	if(IS_DETACHED(iter)) return GrB_NULL_POINTER ;

	iter->min_row = rowIdx ;
	iter->max_row = rowIdx ;

	_set_iter_range(&iter->m_it, iter->min_row, iter->max_row, &iter->m_depleted) ;
	_set_iter_range(&iter->dp_it, iter->min_row, iter->max_row, &iter->dp_depleted) ;

	return GrB_SUCCESS ;
}

GrB_Info Delta_MatrixTupleIter_iterate_range
(
	Delta_MatrixTupleIter *iter,  // iterator to use
	GrB_Index startRowIdx,        // row index to start with
	GrB_Index endRowIdx           // row index to finish with
) {
	if(IS_DETACHED(iter)) return GrB_NULL_POINTER ;

	iter->min_row = startRowIdx ;
	iter->max_row = endRowIdx ;

	_set_iter_range(&iter->m_it, iter->min_row, iter->max_row, &iter->m_depleted) ;
	_set_iter_range(&iter->dp_it, iter->min_row, iter->max_row, &iter->dp_depleted) ;

	return GrB_SUCCESS ;
}

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

// iterate over M matrix
static GrB_Info _next_m_iter_bool
(
	Delta_MatrixTupleIter *iter,  // iterator scanning M
	const GrB_Matrix DM,          // delta-minus, masked entries
	GrB_Index *row,               // optional extracted row index
	GrB_Index *col,               // optional extracted column index
	bool *val,                    // optional extracted value
	bool *depleted                // [output] true if iterator depleted
) {
	ASSERT(iter     != NULL) ;
	ASSERT(DM       != NULL) ;
	ASSERT(depleted != NULL) ;

	GrB_Index  _row ;
	GrB_Index  _col ;
	GrB_Index  _val;

	GxB_Iterator m_it = &iter->m_it ;

	do {
		// iterator depleted, return
		if(*depleted) return GrB_NO_VALUE ;

		_row = GxB_rowIterator_getRowIndex (m_it) ;
		_col = GxB_rowIterator_getColIndex (m_it) ;
		_val = GxB_Iterator_get_BOOL (m_it) ;

		// prep value for next iteration
		_iter_next(m_it, iter->max_row, depleted);

		// if entry is zombie, (val is false) continue
 		if(_val != BOOL_ZOMBIE) break ; // entry isn't deleted, return
	} while (true) ;

	if(row) *row = _row ;
	if(col) *col = _col ;
	if(val) *val = _val ;

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

	GrB_Info             info     =  GrB_SUCCESS                    ;
	GrB_Matrix           DM       =  DELTA_MATRIX_DELTA_MINUS(iter->A) ;
	GxB_Iterator         dp_it    =  &iter->dp_it                   ;

	if(!iter->m_depleted) {
		info = _next_m_iter_bool(iter, DM, row, col, val, &iter->m_depleted) ;
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
	const GrB_Matrix DM,          // delta-minus, masked entries
	GrB_Index *row,               // optional extracted row index
	GrB_Index *col,               // optional extracted column index
	uint64_t *val,                // optional extracted value
	bool *depleted                // [output] true if iterator depleted
) {
	ASSERT(iter     != NULL) ;
	ASSERT(DM       != NULL) ;
	ASSERT(depleted != NULL) ;

	GrB_Index    _row ;
	GrB_Index    _col ;
	uint64_t     _val ;
	GxB_Iterator m_it = &iter->m_it ;

	do {
		// iterator depleted, return
		if(*depleted) return GrB_NO_VALUE;

		_row = GxB_rowIterator_getRowIndex (m_it) ;
		_col = GxB_rowIterator_getColIndex (m_it) ;
		_val = GxB_Iterator_get_UINT64 (m_it) ;

		// prep value for next iteration
		_iter_next(m_it, iter->max_row, depleted) ;

		//check if entry is deleted, if not, return.
		if(_val != U64_ZOMBIE) break ;
	} while (true) ;

	if(row) *row = _row ;
	if(col) *col = _col ;
	if(val) *val = _val ;
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
	GrB_Matrix    DM     =  DELTA_MATRIX_DELTA_MINUS(iter->A) ;
	GxB_Iterator  dp_it  =  &iter->dp_it                    ;

	if(!iter->m_depleted) {
		info = _next_m_iter_uint64(iter, DM, row, col, val, &iter->m_depleted) ;
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

// reset iterator, assumes the iterator is valid
GrB_Info Delta_MatrixTupleIter_reset
(
	Delta_MatrixTupleIter *iter  // iterator to reset
) {
	GrB_Info info = GrB_SUCCESS;

	if(IS_DETACHED(iter)) return GrB_NULL_POINTER ;

	_set_iter_range(&iter->m_it, iter->min_row, iter->max_row, &iter->m_depleted) ;
	_set_iter_range(&iter->dp_it, iter->min_row, iter->max_row, &iter->dp_depleted) ;

	return info ;
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
	if(A == NULL) return GrB_NULL_POINTER ;
	if(iter == NULL) return GrB_NULL_POINTER ;

	GrB_Matrix M  = DELTA_MATRIX_M(A) ;
	GrB_Matrix DP = DELTA_MATRIX_DELTA_PLUS(A) ;

	iter->A = A ;
	iter->min_row = min_row ;
	iter->max_row = max_row ;

	_init_iter(&iter->m_it, M, iter->min_row, iter->max_row, &iter->m_depleted) ;
	_init_iter(&iter->dp_it, DP, iter->min_row, iter->max_row, &iter->dp_depleted) ;

	return GrB_SUCCESS ;
}

// free iterator data
GrB_Info Delta_MatrixTupleIter_detach
(
	Delta_MatrixTupleIter *iter  // iterator to free
) {
	ASSERT(iter != NULL) ;

	iter->A           = NULL ;
	iter->m_depleted  = true ;
	iter->dp_depleted = true ;

	return GrB_SUCCESS ;
}
