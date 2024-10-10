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


//------------------------------------------------------------------------------
// tensor iterator
//------------------------------------------------------------------------------

// tensor iterator
// iterates over a 3D matrix
typedef struct TensorIterator TensorIterator;

struct TensorIterator {
	char private[904];
};

// tensor iterator iteration strategy
// available strategies:
// 1. depleted         - no entries
// 2. scalar           - single entry
// 3. vector           - list of entries
// 4. range of vectors - list of vectors
typedef bool (*IterFunc)(TensorIterator *, GrB_Index*, GrB_Index*, uint64_t*);

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
	uint64_t *x          // [optional out] edge id
);

// checks whether iterator is attached to tensor
bool TensorIterator_is_attached
(
	const TensorIterator *it,  // iterator
	const Tensor T             // tensor
);

