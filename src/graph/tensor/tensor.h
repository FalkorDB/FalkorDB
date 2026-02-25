/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// A tensor is a 3D matrix where A[i,j] is a vector

#include "../entities/edge.h"
#include "../delta_matrix/delta_matrix.h"
#include "../delta_matrix/delta_matrix_iter.h"

// Tensor is a 3D Delta Matrix
typedef Delta_Matrix Tensor;

// checks if x represents scalar entry, if not x is a vector
#define SCALAR_ENTRY(x) !((x) & MSB_MASK)

// clear MSB and cast to GrB_Vector
#define AS_VECTOR(x) (GrB_Vector)(CLEAR_MSB(x));

// init new tensor
Tensor Tensor_new
(
	GrB_Index nrows,  // # rows
	GrB_Index ncols   // # columns
);

// set entry at T[row, col] = x
void Tensor_SetElement
(
	Tensor T,       // tensor
	GrB_Index row,  // row
	GrB_Index col,  // col
	uint64_t x      // value
);

// set multiple entries
void Tensor_SetElements
(
	Tensor T,                        // tensor
	#if defined(__cplusplus)
	const GrB_Index *rows,  // array of row indices
	const GrB_Index *cols,  // array of column indices
	const uint64_t *vals,   // values
	#else
	const GrB_Index *restrict rows,  // array of row indices
	const GrB_Index *restrict cols,  // array of column indices
	const uint64_t *restrict vals,   // values
	#endif
	uint64_t n                       // number of elements
);

// set multiple entries
void Tensor_SetEdges
(
	Tensor T,            // tensor
	const Edge **edges,  // assume edges are sorted by src and dest
	uint64_t n           // number of edges
);

// remove multiple entries
// assuming T's entries are all scalar
void Tensor_RemoveElements_Flat
(
	Tensor T,              // tensor
	const Edge *elements,  // elements to remove
	uint64_t n             // number of elements
);

// remove multiple entries
void Tensor_RemoveElements
(
	Tensor T,                   // tensor
	const Edge *elements,       // elements to remove
	uint64_t n,                 // number of elements
	uint64_t **cleared_entries  // [optional] cleared entries, referes elements
);

// clear all elements specified by A from T
void Tensor_ClearElements
(
	Tensor T,            // tensor to remove entries from
	const GrB_Matrix A,  // elements to remove
	const GrB_Matrix AT  // A's transpose
);

// computes row degree of T[row:]
uint64_t Tensor_RowDegree
(
	Tensor T,      // tensor
	GrB_Index row  // row
);

// computes col degree of T[:col]
uint64_t Tensor_ColDegree
(
	const Tensor T,  // tensor
	GrB_Index col    // col
);

// free tensor
void Tensor_free
(
	Tensor *T  // tensor
);


//------------------------------------------------------------------------------
// tensor iterator
//------------------------------------------------------------------------------

// tensor iterator
// iterates over a 3D matrix
typedef struct TensorIterator TensorIterator;

// tensor iterator iteration strategy
// available strategies:
// 1. depleted         - no entries
// 2. scalar           - single entry
// 3. vector           - list of entries
// 4. range of vectors - list of vectors
typedef bool (*IterFunc)(
		TensorIterator *,
		GrB_Index*,
		GrB_Index*,
		uint64_t*,
		bool*);

// tensor iterator
struct TensorIterator {
	Tensor T;                        // iterated tensor
	Delta_MatrixTupleIter a_it;      // vectors iterator
	struct GB_Iterator_opaque v_it;  // vector iterator
	bool vec;                        // iterate using v_it
	uint64_t x;                      // current entry value
	GrB_Index row;                   // current row
	GrB_Index col;                   // current col
	IterFunc iter_func;              // iteration strategy
};

// iterate vector at T[row, col]
void TensorIterator_ScanEntry
(
	TensorIterator *it,  // iterator
	Tensor T,            // tensor
	GrB_Index row,       // row
	GrB_Index col        // column
);

// iterate over a range of vectors
// scans tensor from M[min:] up to and including M[max:]
void TensorIterator_ScanRange
(
	TensorIterator *it,  // iterator
	Tensor T,            // tensor
	GrB_Index min_row,   // minimum row
	GrB_Index max_row,   // maximum row
	bool transpose       // scan transposed of T
);

// advance iterator
bool TensorIterator_next
(
	TensorIterator *it,  // iterator
	GrB_Index *row,      // [optional out] source id
	GrB_Index *col,      // [optional out] dest id
	uint64_t *x,         // [optional out] edge id
	bool *tensor         // [optional out] tensor
);

// checks whether iterator is attached to tensor
bool TensorIterator_is_attached
(
	const TensorIterator *it,  // iterator
	const Tensor T             // tensor
);

