/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "relation_iterator.h"
#include "delta_matrix/delta_matrix_iter.h"

//------------------------------------------------------------------------------
// advance strategies
//------------------------------------------------------------------------------

// depleted iterator
static bool _DepletedIter
(
	TensorIterator *it,  // iterator
	GrB_Index *row,      // [optional] row
	GrB_Index *col,      // [optional] col
	uint64_t *x          // [optional] value
) {
	ASSERT(it != NULL);

	return false;
}

// scalar iterator
// consumes a single scalar entry and deplete
static bool _ScalarIter
(
	TensorIterator *it,  // iterator
	GrB_Index *row,      // [optional] row
	GrB_Index *col,      // [optional] col
	uint64_t *x          // [optional] value
) {
	ASSERT(it != NULL);

	if(x)   *x   = it->x;
	if(row) *row = it->row;
	if(col) *col = it->col;

	// set iterator as depleted
	it->iter_func = _DepletedIter;

	return true;
}

// vector iterator
// consumes an entire vector and deplete
static bool _VectorIter
(
	TensorIterator *it,  // iterator
	GrB_Index *row,      // [optional] row
	GrB_Index *col,      // [optional] col
	uint64_t *x          // [optional] value
) {
	ASSERT(it != NULL);

	GrB_Info info = Delta_MatrixTupleIter_next_BOOL(&it->v_it, row, x, col);

	// depleted
	if(info != GrB_SUCCESS) {
		it->iter_func = _DepletedIter;
	}

	return info == GrB_SUCCESS;
}

// iterate over a tensor scanning a range of vectors entry by entry
static bool _RangeIter
(
	TensorIterator *it,  // iterator
	GrB_Index *row,      // [optional] row
	GrB_Index *col,      // [optional] col
	uint64_t *x          // [optional] value
) {
	ASSERT(it != NULL);

	GrB_Info info;

	// resuming scan over vector
	if(Delta_MatrixTupleIter_is_attached(&it->v_it, it->T->V)) {
		info = Delta_MatrixTupleIter_next_BOOL(&it->v_it, NULL, &it->x, NULL);
		if(info == GrB_SUCCESS) {
			// consume vector entry
			if(x)   *x   = it->x;
			if(row) *row = it->row;
			if(col) *col = it->col;
			return true;
		}

		// vector depleted, detach vector iterator
		Delta_MatrixTupleIter_detach(&it->v_it);
	}

	// trying to advance to the next vector
	info = Delta_MatrixTupleIter_next_UINT64(&it->a_it, &it->row, &it->col,
			&it->x);
	if(info == GrB_SUCCESS) {
		if(!SCALAR_ENTRY(it->x)) {
			// vector entry, attach iterator and get the first element
			GrB_Index _x = CLEAR_MSB(it->x);
			Delta_MatrixTupleIter_AttachRange(&it->v_it, it->M->V, _x, _x);
			info = Delta_MatrixTupleIter_next_BOOL(&it->v_it, NULL, &it->x, NULL);
			ASSERT(info == GrB_SUCCESS);
		}
		
		// set outputs
		if(x)   *x   = it->x;
		if(row) *row = it->row;
		if(col) *col = it->col;

		return true;
	}

	// no more entries, iterator depeleted
	it->iter_func = _DepletedIter;
	return false;
}

// iterate over a transposed tensor scanning a range of vectors entry by entry
static bool _TransposeRangeIter
(
	TensorIterator *it,  // iterator
	GrB_Index *row,      // [optional] row
	GrB_Index *col,      // [optional] col
	uint64_t *x          // [optional] value
) {
	ASSERT(it != NULL);

	GrB_Index _row;
	GrB_Index _col;
	bool res = _RangeIter(it, _row, _col, x);

	// swap row and col
	if(row) *row = _col;
	if(col) *col = _row;

	return res;
}

//------------------------------------------------------------------------------

// iterate vector at T[row, col]
void TensorIterator_ScanEntry
(
	TensorIterator *it,  // iterator
	const Tensor T,      // tensor
	GrB_Index row,       // row
	GrB_Index col        // column
) {
	ASSERT(T  != NULL);
	ASSERT(it != NULL);

	// reset iterator
	memset(it, 0, sizeof(TensorIterator));

	it->T   = T;
	it->row = row;
	it->col = col;

	GrB_Info info = Delta_Matrix_extractElement_UINT64(&it->x, it->T->A, row,
			col);

	if(info == GrB_SUCCESS) {
		if(SCALAR_ENTRY(it->x)) {
			it->iter_func = _ScalarIter;
		} else {
			uint64_t x = CLEAR_MSB(it->x);
			Delta_MatrixTupleIter_AttachRange(&it->v_it, it->T->V, x, x);
			it->iter_func = _VectorIter;
		}
	} else {
		// missing entry, depeleted iterator
		it->iter_func = _DepletedIter;
	}
}

// iterate over a range of vectors
// scans tensor from M[min:] up to and including M[max:]
void TensorIterator_ScanRange
(
	TensorIterator *it,  // iterator
	const Tensor T,      // tensor
	GrB_Index min_row,   // minimum row
	GrB_Index max_row,   // maximum row
	bool transposed      // transpose
) {
	ASSERT(M  != NULL);
	ASSERT(it != NULL);

	// reset iterator
	memset(it, 0, sizeof(TensorIterator));

	it->T = T;
	IterFunc f = _RangeIter
	Delta_Matrix A = T->A;

	if(transposed) {
		f = _TransposeRangeIter;
		A = Delta_Matrix_getTranspose(A);
	}

	Delta_MatrixTupleIter_AttachRange(&it->r_it, A, min_row, max_row);
	it->iter_func = f;
}

// advance iterator
bool TensorIterator_next
(
	TensorIterator *it,  // iterator
	GrB_Index *row,      // [optional out] source id
	GrB_Index *col,      // [optional out] dest id
	uint64_t *x          // [optional out] edge id
) {
	ASSERT(it != NULL);

	return it->iter_func(it, row, col, x);
}

// checks whether iterator is attached to tensor
bool TensorIterator_is_attached
(
	const TensorIterator *it,  // iterator
	const Tensor T             // tensor
) {
	ASSERT(T  != NULL);
	ASSERT(it != NULL);

	return it->T == T;
}

