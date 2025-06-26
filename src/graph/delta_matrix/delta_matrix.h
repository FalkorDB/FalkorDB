/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "RG.h"
#include "GraphBLAS.h"

#include <pthread.h>

// forward declaration of Delta_Matrix type
typedef struct _Delta_Matrix _Delta_Matrix;
typedef _Delta_Matrix *Delta_Matrix;

// Checks if X represents edge ID.
#define SINGLE_EDGE(x) (!((x) & MSB_MASK))

#define DELTA_MATRIX_M(C) (C)->matrix
#define DELTA_MATRIX_DELTA_PLUS(C) (C)->delta_plus
#define DELTA_MATRIX_DELTA_MINUS(C) (C)->delta_minus

#define DELTA_MATRIX_TM(C) (C)->transposed->matrix
#define DELTA_MATRIX_TDELTA_PLUS(C) (C)->transposed->delta_plus
#define DELTA_MATRIX_TDELTA_MINUS(C) (C)->transposed->delta_minus

#define DELTA_MATRIX_MAINTAIN_TRANSPOSE(C) ((C)->transposed != NULL)

#define DELTA_MATRIX_MULTI_EDGE(M) __extension__({ \
	GrB_Type t;                       \
	GrB_Matrix m = DELTA_MATRIX_M(M); \
	GxB_Matrix_type(&t, m);           \
	(t == GrB_UINT64);                \
})

#define U64_ZOMBIE MSB_MASK
#define BOOL_ZOMBIE ((bool) false)

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

struct _Delta_Matrix {
	volatile bool dirty;      // Indicates if matrix requires sync
	GrB_Matrix matrix;        // Underlying GrB_Matrix
	GrB_Matrix delta_plus;    // Pending additions
	GrB_Matrix delta_minus;   // Pending deletions
	Delta_Matrix transposed;  // Transposed matrix
	pthread_mutex_t mutex;    // Lock
};

GrB_Info Delta_Matrix_new
(
	Delta_Matrix *A,  // handle of matrix to create
	GrB_Type type,    // type of matrix to create
	GrB_Index nrows,  // matrix dimension is nrows-by-ncols
	GrB_Index ncols,
	bool transpose
);

// returns transposed matrix of C
Delta_Matrix Delta_Matrix_getTranspose
(
	const Delta_Matrix C
);

// mark matrix as dirty
void Delta_Matrix_setDirty
(
	Delta_Matrix C
);

bool Delta_Matrix_isDirty
(
	const Delta_Matrix C
);

// checks if C is fully synced
// a synced delta matrix does not contains any entries in
// either its delta-plus and delta-minus internal matrices
bool Delta_Matrix_Synced
(
	const Delta_Matrix C  // matrix to inquery
);

// locks the matrix
void Delta_Matrix_lock
(
	Delta_Matrix C
);

// unlocks the matrix
void Delta_Matrix_unlock
(
	Delta_Matrix C
);

GrB_Info Delta_Matrix_nrows
(
	GrB_Index *nrows,
	const Delta_Matrix C
);

GrB_Info Delta_Matrix_ncols
(
	GrB_Index *ncols,
	const Delta_Matrix C
);

// get the number of entries in a matrix
GrB_Info Delta_Matrix_nvals
(
	GrB_Index *nvals,     // matrix has nvals entries
	const Delta_Matrix A  // matrix to query
);

// change the size of a matrix
GrB_Info Delta_Matrix_resize
(
	Delta_Matrix C,       // matrix to modify
	GrB_Index nrows_new,  // new number of rows in matrix
	GrB_Index ncols_new   // new number of columns in matrix
);

// C (i,j) = x
GrB_Info Delta_Matrix_setElement_BOOL
(
	Delta_Matrix C,  // matrix to modify
	GrB_Index i,     // row index
	GrB_Index j      // column index
);

// C (i,j) = x
GrB_Info Delta_Matrix_setElement_UINT64
(
	Delta_Matrix C,  // matrix to modify
	uint64_t x,      // scalar to assign to C(i,j)
	GrB_Index i,     // row index
	GrB_Index j      // column index
);

// x = A(i,j)
GrB_Info Delta_Matrix_extractElement_BOOL     
(
	bool *x,               // extracted scalar
	const Delta_Matrix A,  // matrix to extract a scalar from
	GrB_Index i,           // row index
	GrB_Index j            // column index
) ;

// x = A(i,j)
GrB_Info Delta_Matrix_extractElement_UINT64   
(
	uint64_t *x,           // extracted scalar
	const Delta_Matrix A,  // matrix to extract a scalar from
	GrB_Index i,           // row index
	GrB_Index j            // column index
) ;

// remove entry at position C[i,j]
GrB_Info Delta_Matrix_removeElement_BOOL
(
	Delta_Matrix C,  // matrix to remove entry from
	GrB_Index i,     // row index
	GrB_Index j      // column index
);

GrB_Info Delta_Matrix_removeElement_UINT64
(
	Delta_Matrix C,  // matrix to remove entry from
	GrB_Index i,     // row index
	GrB_Index j      // column index
);

GrB_Info Delta_Matrix_removeElements
(
	Delta_Matrix C,  // matrix to remove entry from
	GrB_Matrix A     // matrix filled with elements to remove
) ;

// C = A * B
GrB_Info Delta_mxm
(
	Delta_Matrix C,               // input/output matrix for results
	const GrB_Semiring semiring,  // defines '+' and '*' for A*B
	const Delta_Matrix A,         // first input:  matrix A
	const Delta_Matrix B          // second input: matrix B
);

// Does not look at dm. Assumes that any "zombie" value is '0'
// where x \otimes 0 = 0' and x + 0' = x. (AKA the semiring "zero")
// NOTE: this does not remove explicit zombies.
// To make the output matrix a proper delta matrix, either remove the zombies 
// or make dm contain all entries that are zombies.
// C = A * B
GrB_Info Delta_mxm_identity                    
(
    GrB_Matrix C,                 // input/output matrix for results may contain zombie values
    const GrB_Semiring semiring,  // defines '+' and '*' for A*B
    const GrB_Matrix A,           // first input:  matrix A may contain zombie values
    const Delta_Matrix B          // second input: matrix B
);

// Inputs need not be synced, but must be of the same type.
// Output is a valid delta matrix, which may not be synced.
// zombies must be the identity of the given monoid.
// C = A + B
GrB_Info Delta_eWiseAdd
(
    Delta_Matrix C,       // input/output matrix for results
    const GrB_Monoid op,  // defines '+' for T=A+B
    const Delta_Matrix A, // first input:  matrix A
    const Delta_Matrix B  // second input: matrix B
);

// clear a matrix of all entries
GrB_Info Delta_Matrix_clear  
(                            // type and dimensions remain unchanged
    Delta_Matrix A           // matrix to clear
);

// copy matrix A to matrix C
GrB_Info Delta_Matrix_copy 
(
	Delta_Matrix C,       // output matrix
	const Delta_Matrix A  // input matrix
);

// get matrix C without writing to internal matrix
GrB_Info Delta_Matrix_export
(
	GrB_Matrix *A,
	Delta_Matrix C
) ;

// get structural matrix A without writing to internal matrix
GrB_Info Delta_Matrix_export_structure
(
	GrB_Matrix *A,
	Delta_Matrix C
) ;

// checks to see if matrix has pending operations
GrB_Info Delta_Matrix_pending
(
	const Delta_Matrix C,  // matrix to query
	bool *pending          // are there any pending operations
);

GrB_Info Delta_Matrix_wait
(
	Delta_Matrix C,
	bool force_sync
);

// get the type of the M matrix
GrB_Info Delta_Matrix_type
(
	GrB_Type *type,
	Delta_Matrix A
);

void Delta_Matrix_free
(
	Delta_Matrix *C
);

const GrB_Matrix Delta_Matrix_M
(
	const Delta_Matrix C
);

// return # of bytes used for a matrix
GrB_Info Delta_Matrix_memoryUsage
(
    size_t *size,         // # of bytes used by the matrix A
    const Delta_Matrix A  // matrix to query
);

void Delta_Matrix_synchronize
(
	Delta_Matrix C,
	GrB_Index nrows,
	GrB_Index ncols
);

// replace C's internal M matrix with given M
// the operation can only succeed if C's interal matrices:
// M, DP, DM are all empty
// C->M will point to *M and *M will be set to NULL
GrB_Info Delta_Matrix_setM
(
	Delta_Matrix C,  // delta matrix
	GrB_Matrix *M    // new M
);