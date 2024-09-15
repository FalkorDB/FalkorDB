/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// tensor iterator
// iterates over a 3D matrix
typedef struct TensorIterator TensorIterator;

// tensor iterator iteration strategy
// available strategies:
// 1. depleted         - no entries
// 2. scalar           - single entry
// 3. vector           - list of entries
// 4. range of vectors - list of vectors
typedef bool (*IterFunc)(TensorIterator *, GrB_Index*, GrB_Index*, uint64_t*);

// tensor iterator
struct TensorIterator {
	const Tensor T;              // iterated tensor
	Delta_MatrixTupleIter a_it;  // vectors iterator
	Delta_MatrixTupleIter v_it;  // vector iterator
	uint64_t x;                  // current entry value
	GrB_Index row;               // current row
	GrB_Index col;               // current col
	IterFunc iter_func;          // iteration strategy
};

// iterate vector at T[row, col]
void TensorIterator_ScanEntry
(
	TensorIterator *it,  // iterator
	const Tensor T,      // tensor
	GrB_Index row,       // row
	GrB_Index col        // column
);

// iterate over a range of vectors
// scans tensor from M[min:] up to and including M[max:]
void TensorIterator_ScanRange
(
	TensorIterator *it,  // iterator
	const Tensor T,      // tensor
	GrB_Index min_row,   // minimum row
	GrB_Index max_row,   // maximum row
	bool transposed      // transpose
);

// advance iterator
bool TensorIterator_next
(
	TensorIterator *it,  // iterator
	GrB_Index *row,      // [optional out] source id
	GrB_Index *col,      // [optional out] dest id
	uint64_t *x          // [optional out] edge id
);

// checks whether iterator is attached to tensor
bool TensorIterator_is_attached
(
	const TensorIterator *it,  // iterator
	const Tensor T             // tensor
);

