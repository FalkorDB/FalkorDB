/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdint.h>
#include "./delta_matrix.h"
#include "GraphBLAS.h"

#define DELTA_ITER_MIN_ROW 0
#define DELTA_ITER_MAX_ROW ULLONG_MAX

// M's row iterator combined with the pending deletions (DM) that mask it:
// advancing/seeking this always transparently skips any M entry that's been
// logically deleted, so its current position - whenever `depleted` is false -
// is guaranteed a live entry. Callers never need to consult DM themselves.
typedef struct
{
	struct GB_Iterator_opaque it;     // M's row iterator
	struct GB_Iterator_opaque dm_it;  // DM's row iterator (the mask)
	bool depleted;                    // is it depleted
	bool dm_depleted;                 // is dm_it depleted
} Delta_MaskedMIter ;

// TuplesIter maintains information required
// to iterate over a Delta_Matrix
typedef struct
{
	Delta_Matrix A;                   // matrix iterated
	Delta_MaskedMIter m;              // main matrix, masked by pending deletions
	struct GB_Iterator_opaque dp_it;  // internal delta plus iterator
	bool dp_depleted;                 // is dp iterator depleted
	GrB_Index min_row;                // minimum row for iteration
	GrB_Index max_row;                // maximum row for iteration
} Delta_MatrixTupleIter ;

// attach iterator to matrix
GrB_Info Delta_MatrixTupleIter_attach
(
	Delta_MatrixTupleIter *iter,  // iterator to update
	const Delta_Matrix A          // matrix to scan
);

// attach iterator to matrix governing the specified range
GrB_Info Delta_MatrixTupleIter_AttachRange
(
	Delta_MatrixTupleIter *iter,  // iterator to update
	const Delta_Matrix A,         // matrix to scan
	GrB_Index min_row,            // minimum row for iteration
	GrB_Index max_row             // maximum row for iteration
);

// free iterator internals, keeping the iterator intact
GrB_Info Delta_MatrixTupleIter_detach
(
	Delta_MatrixTupleIter *iter  // iterator to free
);

// returns true if iterator is attached to given matrix false otherwise
bool Delta_MatrixTupleIter_is_attached
(
	const Delta_MatrixTupleIter *iter,  // iterator to check
	const Delta_Matrix M                // matrix attached to
);

// iterate over a single row
GrB_Info Delta_MatrixTupleIter_iterate_row
(
	Delta_MatrixTupleIter *iter,   // iterator to use
	GrB_Index rowIdx               // row to iterate
);

// iterate over a range of rows
GrB_Info Delta_MatrixTupleIter_iterate_range
(
	Delta_MatrixTupleIter *iter,  // iterator to use
	GrB_Index startRowIdx,        // row index to start with
	GrB_Index endRowIdx           // row index to finish with
);

// advance iterator
GrB_Info Delta_MatrixTupleIter_next_BOOL
(
	Delta_MatrixTupleIter *iter,  // iterator to consume
	GrB_Index *row,               // optional output row index
	GrB_Index *col,               // optional output column index
	bool *val                     // optional value at A[row, col]
);

GrB_Info Delta_MatrixTupleIter_next_UINT64
(
	Delta_MatrixTupleIter *iter,  // iterator to consume
	GrB_Index *row,               // optional output row index
	GrB_Index *col,               // optional output column index
	uint64_t *val                 // optional value at A[row, col]
);

// advance iterator in true ascending (row, col) order across the whole
// attached range, merging M (pending deletions masked out) and delta-plus
// instead of draining M fully before ever yielding from delta-plus. Use
// this instead of Delta_MatrixTupleIter_next_BOOL when a caller resumes a
// range scan across multiple attach/detach cycles (e.g. batched index
// population) and needs the guarantee that nothing at or before the last
// row visited remains unvisited.
GrB_Info Delta_MatrixTupleIter_next_BOOL_sorted
(
	Delta_MatrixTupleIter *iter,  // iterator to consume
	GrB_Index *row,               // optional output row index
	GrB_Index *col,               // optional output column index
	bool *val                     // optional value at A[row, col]
);

// UINT64 counterpart of Delta_MatrixTupleIter_next_BOOL_sorted
GrB_Info Delta_MatrixTupleIter_next_UINT64_sorted
(
	Delta_MatrixTupleIter *iter,  // iterator to consume
	GrB_Index *row,               // optional output row index
	GrB_Index *col,               // optional output column index
	uint64_t *val                 // optional value at A[row, col]
);

// reset iterator
GrB_Info Delta_MatrixTupleIter_reset
(
	Delta_MatrixTupleIter *iter  // iterator to reset
);

// return the position of the current iterator entry
// unique and stable across re-scans of the same matrix
GrB_Index Delta_Matrix_Iterator_getp
(
	Delta_MatrixTupleIter *iter  // iterator to query
);

