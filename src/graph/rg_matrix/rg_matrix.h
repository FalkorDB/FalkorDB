/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "RG.h"
#include "GraphBLAS.h"

#include <pthread.h>

// forward declaration of RG_Matrix type
typedef struct _RG_Matrix _RG_Matrix;
typedef _RG_Matrix *RG_Matrix;

// Checks if X represents edge ID.
#define SINGLE_EDGE(x) !((x) & MSB_MASK)

#define RG_MATRIX_M(C) (C)->matrix
#define RG_MATRIX_DELTA_PLUS(C) (C)->delta_plus
#define RG_MATRIX_DELTA_MINUS(C) (C)->delta_minus

#define RG_MATRIX_TM(C) (C)->transposed->matrix
#define RG_MATRIX_TDELTA_PLUS(C) (C)->transposed->delta_plus
#define RG_MATRIX_TDELTA_MINUS(C) (C)->transposed->delta_minus

#define RG_MATRIX_MAINTAIN_TRANSPOSE(C) (C)->transposed != NULL

#define RG_MATRIX_MULTI_EDGE(M) __extension__({ \
	GrB_Type t;                    \
	GrB_Matrix m = RG_MATRIX_M(M); \
	GxB_Matrix_type(&t, m);        \
	(t == GrB_UINT64);             \
})


//------------------------------------------------------------------------------
//
// possible combinations
//
//------------------------------------------------------------------------------
//
//  empty
//
//   A         DP        DM
//   . . .     . . .     . . .
//   . . .     . . .     . . .
//   . . .     . . .     . . .
//
//------------------------------------------------------------------------------
//
//  flushed, no pending changes
//
//   A         DP        DM
//   . 1 .     . . .     . . .
//   . . .     . . .     . . .
//   . . .     . . .     . . .
//
//------------------------------------------------------------------------------
//
//  single entry added
//
//   A         DP        DM
//   . . .     . 1 .     . . .
//   . . .     . . .     . . .
//   . . .     . . .     . . .
//
//------------------------------------------------------------------------------
//
//  single entry deleted
//
//   A         DP        DM
//   1 . .     . . .     1 . .
//   . . .     . . .     . . .
//   . . .     . . .     . . .
//
//------------------------------------------------------------------------------
//  impossible state
//  existing entry deleted and then added back
//
//   A         DP        DM
//   1 . .     1 . .     1 . .
//   . . .     . . .     . . .
//   . . .     . . .     . . .
//
//------------------------------------------------------------------------------
//
//  impossible state
//  marked none existing entry for deletion
//
//   A         DP        DM
//   . . .     . . .     1 . .
//   . . .     . . .     . . .
//   . . .     . . .     . . .
//
//------------------------------------------------------------------------------
//
//  impossible state
//  adding to an already existing entry
//  should have turned A[0,0] to a multi-value
//
//   A         DP        DM
//   1 . .     1 . .     . . .
//   . . .     . . .     . . .
//   . . .     . . .     . . .
//
//------------------------------------------------------------------------------
//
//  impossible state
//  deletion of pending entry should have cleared it DP[0,0]
//
//   A         DP        DM
//   . . .     1 . .     1 . .
//   . . .     . . .     . . .
//   . . .     . . .     . . .
//
//------------------------------------------------------------------------------

struct _RG_Matrix {
	volatile bool dirty;                // Indicates if matrix requires sync
	GrB_Matrix matrix;                  // Underlying GrB_Matrix
	GrB_Matrix delta_plus;              // Pending additions
	GrB_Matrix delta_minus;             // Pending deletions
	RG_Matrix transposed;               // Transposed matrix
	pthread_mutex_t mutex;              // Lock
};

GrB_Info RG_Matrix_new
(
	RG_Matrix *A,            // handle of matrix to create
	GrB_Type type,           // type of matrix to create
	GrB_Index nrows,         // matrix dimension is nrows-by-ncols
	GrB_Index ncols
);

// returns transposed matrix of C
RG_Matrix RG_Matrix_getTranspose
(
	const RG_Matrix C
);

// mark matrix as dirty
void RG_Matrix_setDirty
(
	RG_Matrix C
);

bool RG_Matrix_isDirty
(
	const RG_Matrix C
);

// checks if C is fully synced
// a synced delta matrix does not contains any entries in
// either its delta-plus and delta-minus internal matrices
bool RG_Matrix_Synced
(
	const RG_Matrix C  // matrix to inquery
);

// locks the matrix
void RG_Matrix_Lock
(
	RG_Matrix C
);

// unlocks the matrix
void RG_Matrix_Unlock
(
	RG_Matrix C
);

GrB_Info RG_Matrix_nrows
(
	GrB_Index *nrows,
	const RG_Matrix C
);

GrB_Info RG_Matrix_ncols
(
	GrB_Index *ncols,
	const RG_Matrix C
);

GrB_Info RG_Matrix_nvals    // get the number of entries in a matrix
(
	GrB_Index *nvals,       // matrix has nvals entries
	const RG_Matrix A       // matrix to query
);

GrB_Info RG_Matrix_resize      // change the size of a matrix
(
	RG_Matrix C,                // matrix to modify
	GrB_Index nrows_new,        // new number of rows in matrix
	GrB_Index ncols_new         // new number of columns in matrix
);

GrB_Info RG_Matrix_setElement_BOOL      // C (i,j) = x
(
	RG_Matrix C,                        // matrix to modify
	GrB_Index i,                        // row index
	GrB_Index j                         // column index
);

GrB_Info RG_Matrix_setElement_UINT64      // C (i,j) = x
(
	RG_Matrix C,                        // matrix to modify
	uint64_t x,                         // scalar to assign to C(i,j)
	GrB_Index i,                        // row index
	GrB_Index j                         // column index
);

GrB_Info RG_Matrix_extractElement_BOOL     // x = A(i,j)
(
	bool *x,                               // extracted scalar
	const RG_Matrix A,                     // matrix to extract a scalar from
	GrB_Index i,                           // row index
	GrB_Index j                            // column index
) ;

GrB_Info RG_Matrix_extractElement_UINT64   // x = A(i,j)
(
	uint64_t *x,                           // extracted scalar
	const RG_Matrix A,                     // matrix to extract a scalar from
	GrB_Index i,                           // row index
	GrB_Index j                            // column index
) ;

// remove entry at position C[i,j]
GrB_Info RG_Matrix_removeElement_BOOL
(
	RG_Matrix C,                    // matrix to remove entry from
	GrB_Index i,                    // row index
	GrB_Index j                     // column index
);

GrB_Info RG_Matrix_removeElement_UINT64
(
	RG_Matrix C,                    // matrix to remove entry from
	GrB_Index i,                    // row index
	GrB_Index j                     // column index
);

// remove value 'v' from multi-value entry at position C[i,j]
GrB_Info RG_Matrix_removeEntry_UINT64
(
	RG_Matrix C,                    // matrix to remove entry from
	GrB_Index i,                    // row index
	GrB_Index j,                    // column index
	uint64_t  v,                    // value to remove
	bool     *entry_deleted         // is entry deleted
);

GrB_Info RG_mxm                     // C = A * B
(
	RG_Matrix C,                    // input/output matrix for results
	const GrB_Semiring semiring,    // defines '+' and '*' for A*B
	const RG_Matrix A,              // first input:  matrix A
	const RG_Matrix B               // second input: matrix B
);

GrB_Info RG_eWiseAdd                // C = A + B
(
    RG_Matrix C,                    // input/output matrix for results
    const GrB_Semiring semiring,    // defines '+' for T=A+B
    const RG_Matrix A,              // first input:  matrix A
    const RG_Matrix B               // second input: matrix B
);

GrB_Info RG_Matrix_clear    // clear a matrix of all entries;
(                           // type and dimensions remain unchanged
    RG_Matrix A             // matrix to clear
);

GrB_Info RG_Matrix_copy     // copy matrix A to matrix C
(
	RG_Matrix C,            // output matrix
	const RG_Matrix A       // input matrix
);

// get matrix C without writing to internal matrix
GrB_Info RG_Matrix_export
(
	GrB_Matrix *A,
	RG_Matrix C
);

// checks to see if matrix has pending operations
GrB_Info RG_Matrix_pending
(
	const RG_Matrix C,              // matrix to query
	bool *pending                   // are there any pending operations
);

GrB_Info RG_Matrix_wait
(
	RG_Matrix C,
	bool force_sync
);

// get the type of the M matrix
GrB_Info RG_Matrix_type
(
	GrB_Type *type,
	RG_Matrix A
);

void RG_Matrix_free
(
	RG_Matrix *C
);

