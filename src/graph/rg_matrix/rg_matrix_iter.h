/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include <stdint.h>
#include "./rg_matrix.h"
#include "../../deps/GraphBLAS/Include/GraphBLAS.h"

#define RG_ITER_MIN_ROW 0
#define RG_ITER_MAX_ROW ULLONG_MAX

// TuplesIter maintains information required
// to iterate over a RG_Matrix
typedef struct
{
	RG_Matrix A;                      // matrix iterated
	struct GB_Iterator_opaque m_it;   // internal m iterator
	struct GB_Iterator_opaque dp_it;  // internal delta plus iterator
	bool m_depleted;                  // is m iterator depleted
	bool dp_depleted;                 // is dp iterator depleted
	GrB_Index min_row;                // minimum row for iteration
	GrB_Index max_row;                // maximum row for iteration
} RG_MatrixTupleIter ;

// attach iterator to matrix
GrB_Info RG_MatrixTupleIter_attach
(
	RG_MatrixTupleIter *iter,       // iterator to update
	const RG_Matrix A               // matrix to scan
);

// attach iterator to matrix governing the specified range
GrB_Info RG_MatrixTupleIter_AttachRange
(
	RG_MatrixTupleIter *iter,       // iterator to update
	const RG_Matrix A,              // matrix to scan
	GrB_Index min_row,              // minimum row for iteration
	GrB_Index max_row               // maximum row for iteration
);

// free iterator internals, keeping the iterator intact
GrB_Info RG_MatrixTupleIter_detach
(
	RG_MatrixTupleIter *iter       // iterator to free
);

// returns true if iterator is attached to given matrix false otherwise
bool RG_MatrixTupleIter_is_attached
(
	const RG_MatrixTupleIter *iter,       // iterator to check
	const RG_Matrix M                     // matrix attached to
);

GrB_Info RG_MatrixTupleIter_iterate_row
(
	RG_MatrixTupleIter *iter,      // iterator to use
	GrB_Index rowIdx               // row to iterate
);

GrB_Info RG_MatrixTupleIter_iterate_range
(
	RG_MatrixTupleIter *iter,   // iterator to use
	GrB_Index startRowIdx,      // row index to start with
	GrB_Index endRowIdx         // row index to finish with
);

// advance iterator
GrB_Info RG_MatrixTupleIter_next_BOOL
(
	RG_MatrixTupleIter *iter,       // iterator to consume
	GrB_Index *row,                 // optional output row index
	GrB_Index *col,                 // optional output column index
	bool *val                       // optional value at A[row, col]
);

GrB_Info RG_MatrixTupleIter_next_UINT64
(
	RG_MatrixTupleIter *iter,       // iterator to consume
	GrB_Index *row,                 // optional output row index
	GrB_Index *col,                 // optional output column index
	uint64_t *val                   // optional value at A[row, col]
);

// reset iterator
GrB_Info RG_MatrixTupleIter_reset
(
	RG_MatrixTupleIter *iter       // iterator to reset
);

