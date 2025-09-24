/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "tensor.h"

//------------------------------------------------------------------------------
// advance strategies
//------------------------------------------------------------------------------

// depleted iterator
static bool _DepletedIter
(
	TensorIterator *it,  // iterator
	GrB_Index *row,      // [optional] row
	GrB_Index *col,      // [optional] col
	uint64_t *x,         // [optional] value
	bool *tensor         // [optional out] tensor
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
	uint64_t *x,         // [optional] value
	bool *tensor         // [optional out] tensor
) {
	ASSERT(it != NULL);

	if(x)      *x      = it->x;
	if(row)    *row    = it->row;
	if(col)    *col    = it->col;
	if(tensor) *tensor = !SCALAR_ENTRY(it->x);

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
	uint64_t *x,         // [optional] value
	bool *tensor         // [optional out] tensor
) {
	ASSERT(it != NULL);

	GxB_Iterator v_it = &it->v_it;

	if(x)      *x      = GxB_Vector_Iterator_getIndex(v_it);
	if(row)    *row    = it->row;
	if(col)    *col    = it->col;
	if(tensor) *tensor = true;

	// advance to next entry
	GrB_Info info = GxB_Vector_Iterator_next(v_it);

	// depleted
	if(info == GxB_EXHAUSTED) {
		it->iter_func = _DepletedIter;
	}

	return true;
}

// iterate over a tensor scanning a range of vectors entry by entry
static bool _TransposeRangeIter
(
	TensorIterator *it,  // iterator
	GrB_Index *row,      // [optional] row
	GrB_Index *col,      // [optional] col
	uint64_t *x,         // [optional] value
	bool *tensor         // [optional out] tensor
) {
	ASSERT(it != NULL);

	GrB_Info info;
	GxB_Iterator v_it;

	// resuming scan over vector
	if(it->vec) {
vector_consume:
		// consume vector entry
		v_it = &it->v_it;

		if(x)      *x      = GxB_Vector_Iterator_getIndex(v_it);
		if(row)    *row    = it->row;
		if(col)    *col    = it->col;
		if(tensor) *tensor = true;

		// preparing next call
		info = GxB_Vector_Iterator_next(v_it);
		if(info == GxB_EXHAUSTED) {
			// vector depleted, detach vector iterator
			it->vec = false;
		}

		return true;
	}

	// iterate over T transpose
	// trying to advance to the next vector
	info = Delta_MatrixTupleIter_next_BOOL(&it->a_it, &it->col, &it->row, NULL);

	if(info == GrB_SUCCESS) {
		// extract T[row, col]
		info = Delta_Matrix_extractElement_UINT64(&it->x, it->T, it->row, it->col);
		ASSERT(info == GrB_SUCCESS);

		if(SCALAR_ENTRY(it->x)) {
			// set outputs
			if(x)      *x      = it->x;
			if(row)    *row    = it->row;
			if(col)    *col    = it->col;
			if(tensor) *tensor = false;

			return true;
		}

		// vector entry, attach iterator and get the first element
		it->vec = true;

		GrB_Vector V = AS_VECTOR(it->x);
		v_it = &it->v_it;

		info = GxB_Vector_Iterator_attach(v_it, V, NULL);
		ASSERT(info == GrB_SUCCESS);

		info = GxB_Vector_Iterator_seek(v_it, 0);
		ASSERT(info == GrB_SUCCESS);

		goto vector_consume;
	}

	// no more entries, iterator depeleted
	it->iter_func = _DepletedIter;
	return false;
}

// iterate over a tensor scanning a range of vectors entry by entry
static bool _RangeIter
(
	TensorIterator *it,  // iterator
	GrB_Index *row,      // [optional] row
	GrB_Index *col,      // [optional] col
	uint64_t *x,         // [optional] value
	bool *tensor         // [optional out] tensor
) {
	ASSERT(it != NULL);

	GrB_Info info;
	GxB_Iterator v_it;

	// resuming scan over vector
	if(it->vec) {
vector_consume:
		// consume vector entry
		v_it = &it->v_it;

		if(x)      *x      = GxB_Vector_Iterator_getIndex(v_it);
		if(row)    *row    = it->row;
		if(col)    *col    = it->col;
		if(tensor) *tensor = true;

		// preparing next call
		info = GxB_Vector_Iterator_next(v_it);
		if(info == GxB_EXHAUSTED) {
			// vector depleted, detach vector iterator
			it->vec = false;
		}

		return true;
	}

	// trying to advance to the next vector
	info = Delta_MatrixTupleIter_next_UINT64(&it->a_it, &it->row, &it->col, &it->x);
	if(info == GrB_SUCCESS) {
		if(SCALAR_ENTRY(it->x)) {
			// set outputs
			if(x)      *x      = it->x;
			if(row)    *row    = it->row;
			if(col)    *col    = it->col;
			if(tensor) *tensor = false;

			return true;
		}

		// vector entry, attach iterator and get the first element
		it->vec = true;

		GrB_Vector V = AS_VECTOR(it->x);
		v_it = &it->v_it;

		info = GxB_Vector_Iterator_attach(v_it, V, NULL);
		ASSERT(info == GrB_SUCCESS);

		info = GxB_Vector_Iterator_seek(v_it, 0);
		ASSERT(info == GrB_SUCCESS);

		goto vector_consume;
	}

	// no more entries, iterator depeleted
	it->iter_func = _DepletedIter;
	return false;
}

//------------------------------------------------------------------------------

// iterate vector at T[row, col]
void TensorIterator_ScanEntry
(
	TensorIterator *it,  // iterator
	Delta_Matrix T,      // tensor
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

	GrB_Info info = Delta_Matrix_extractElement_UINT64(&it->x, it->T, row, col);

	if(info == GrB_SUCCESS) {
		if(SCALAR_ENTRY(it->x)) {
			it->iter_func = _ScalarIter;
		} else {
			// iterate over vector
			GrB_Vector   v    = AS_VECTOR(it->x);
			GxB_Iterator v_it = &it->v_it;

			info = GxB_Vector_Iterator_attach(v_it, v, NULL);
			ASSERT(info == GrB_SUCCESS);

			info = GxB_Vector_Iterator_seek(v_it, 0);
			ASSERT(info == GrB_SUCCESS);

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
	Delta_Matrix T,      // tensor
	GrB_Index min_row,   // minimum row
	GrB_Index max_row,   // maximum row
	bool transpose       // scan transposed of T
) {
	ASSERT(T  != NULL);
	ASSERT(it != NULL);

	// reset iterator
	memset(it, 0, sizeof(TensorIterator));

	it->T = T;

	if(transpose) {
		Delta_Matrix TT = Delta_Matrix_getTranspose(T);
		Delta_MatrixTupleIter_AttachRange(&it->a_it, TT, min_row, max_row);
		it->iter_func = _TransposeRangeIter;
	} else {
		Delta_MatrixTupleIter_AttachRange(&it->a_it, it->T, min_row, max_row);
		it->iter_func = _RangeIter;
	}
}

// advance iterator
bool TensorIterator_next
(
	TensorIterator *it,  // iterator
	GrB_Index *row,      // [optional out] source id
	GrB_Index *col,      // [optional out] dest id
	uint64_t *x,         // [optional out] edge id
	bool *tensor         // [optional out] tensor
) {
	ASSERT(it != NULL);

	return it->iter_func(it, row, col, x, tensor);
}

// checks whether iterator is attached to tensor
bool TensorIterator_is_attached
(
	const TensorIterator *it,  // iterator
	const Delta_Matrix T       // tensor
) {
	ASSERT(T  != NULL);
	ASSERT(it != NULL);

	return it->T == T;
}

