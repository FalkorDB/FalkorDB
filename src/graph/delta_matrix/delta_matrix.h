/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "RG.h"
#include "GraphBLAS.h"

#include <pthread.h>

// forward declaration of Delta_Matrix type
typedef struct _Delta_Matrix _Delta_Matrix;
typedef _Delta_Matrix *Delta_Matrix;

#define DELTA_MATRIX_M(C)            ((C)->matrix)
#define DELTA_MATRIX_DELTA_PLUS(C)   ((C)->delta_plus)
#define DELTA_MATRIX_DELTA_MINUS(C)  ((C)->delta_minus)
#define DELTA_MATRIX_TM(C)            ((C)->transposed->matrix)
#define DELTA_MATRIX_TDELTA_PLUS(C)   ((C)->transposed->delta_plus)
#define DELTA_MATRIX_TDELTA_MINUS(C)  ((C)->transposed->delta_minus)

#define DELTA_MATRIX_MAINTAIN_TRANSPOSE(C) ((C)->transposed != NULL)

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
	Delta_Matrix *A,         // handle of matrix to create
	GrB_Type type,           // type of matrix to create
	GrB_Index nrows,         // matrix dimension is nrows-by-ncols
	GrB_Index ncols,
	bool transpose           // if true, create a transpose of the matrix
);

// returns transposed matrix of C
Delta_Matrix Delta_Matrix_getTranspose
(
	const Delta_Matrix C
);

bool Delta_Matrix_isDirty
(
	const Delta_Matrix C
);

// get the internal matrix M
GrB_Matrix Delta_Matrix_M
(
	const Delta_Matrix C
);

// get the internal matrix delta plus
GrB_Matrix Delta_Matrix_DP
(
	const Delta_Matrix C
);

// get the internal matrix delta minus
GrB_Matrix Delta_Matrix_DM
(
	const Delta_Matrix C
);

// set the internal matrix M
// the operation can only succeed if C's interal matrices are all empty
GrB_Info Delta_Matrix_setM
(
	Delta_Matrix C,  // delta matrix
	GrB_Matrix *M    // new M
);

// Set the internal matricies of C
// the operation can only succeed if C's interal matrices are all empty
GrB_Info Delta_Matrix_setMatrices
(
	Delta_Matrix C,  // delta matrix
	GrB_Matrix *M,   // new M
	GrB_Matrix *DP,  // new delta-plus
	GrB_Matrix *DM   // new delta-minus
) ;

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

GrB_Info Delta_Matrix_nvals  // get the number of entries in a matrix
(
	GrB_Index *nvals,        // matrix has nvals entries
	const Delta_Matrix A     // matrix to query
);

GrB_Info Delta_Matrix_resize  // change the size of a matrix
(
	Delta_Matrix C,           // matrix to modify
	GrB_Index nrows_new,      // new number of rows in matrix
	GrB_Index ncols_new       // new number of columns in matrix
);

GrB_Info Delta_Matrix_setElement_BOOL   // C (i,j) = x
(
	Delta_Matrix C,                     // matrix to modify
	GrB_Index i,                        // row index
	GrB_Index j                         // column index
);

GrB_Info Delta_Matrix_setElement_UINT64  // C (i,j) = x
(
	Delta_Matrix C,                      // matrix to modify
	uint64_t x,                          // value
	GrB_Index i,                         // row index
	GrB_Index j                          // column index
);

GrB_Info Delta_Matrix_extractElement_UINT64  // x = A(i,j)
(
	uint64_t *x,                             // extracted scalar
	const Delta_Matrix A,                    // matrix to extract a scalar from
	GrB_Index i,                             // row index
	GrB_Index j                              // column index
) ;

// check if element A(i,j) is stored in the delta matrix
GrB_Info Delta_Matrix_isStoredElement
(
	const Delta_Matrix A,  // matrix to check
	GrB_Index i,           // row index
	GrB_Index j            // column index
) ;

// remove entry at position C[i,j]
GrB_Info Delta_Matrix_removeElement
(
	Delta_Matrix C,  // matrix to remove entry from
	GrB_Index i,     // row index
	GrB_Index j      // column index
);

GrB_Info Delta_Matrix_removeElements
(
	Delta_Matrix C,      // matrix to remove entries from
	const GrB_Matrix A,  // elements to remove
	const GrB_Matrix AT  // A's transpose
);

// C = AB
// A should be fully synced on input
// C will be fully synced on output
GrB_Info Delta_mxm
(
	Delta_Matrix C,               // input/output matrix for results
	const GrB_Semiring semiring,  // defines '+' and '*' for A*B
	const Delta_Matrix A,         // first input:  matrix A (Must be synced)
	const Delta_Matrix B          // second input: matrix B
);

// C = A + B 
// C is fully synced on output
GrB_Info Delta_eWiseAdd
(
    Delta_Matrix C,               // input/output matrix for results
    const GrB_Semiring semiring,  // defines '+' for T=A+B
    const Delta_Matrix A,         // first input:  matrix A
    const Delta_Matrix B          // second input: matrix B
);

GrB_Info Delta_Matrix_clear  // clear a matrix of all entries;
(                            // type and dimensions remain unchanged
    Delta_Matrix A           // matrix to clear
);

// copy matrix A to matrix C
// does not set the transpose
GrB_Info Delta_Matrix_dup
(
	Delta_Matrix *C,      // output matrix
	const Delta_Matrix A  // input matrix
) ;

// get the fully synced GrB_Matrix from Delta_Matrix C without modifying C
GrB_Info Delta_Matrix_export
(
    GrB_Matrix *A,         // output Matrix 
    const Delta_Matrix C,  // input Delta Matrix
    const GrB_Type type    // output matrix type (values will be typecast)
);

// checks to see if matrix has pending operations
// pending is set to true if any of the internal matricies have pending
// operations
GrB_Info Delta_Matrix_pending
(
	const Delta_Matrix C,  // matrix to query
	bool *pending          // are there any pending operations
);

// return # of bytes used for a matrix
GrB_Info Delta_Matrix_memoryUsage
(
    size_t *size,         // # of bytes used by the matrix A
    const Delta_Matrix A  // matrix to query
);

GrB_Info Delta_Matrix_wait
(
	Delta_Matrix C,
	bool force_sync
);

void Delta_Matrix_synchronize
(
	Delta_Matrix C,
	GrB_Index nrows,
	GrB_Index ncols
);

bool Delta_Matrix_Synced
(
	const Delta_Matrix C
);

void Delta_Matrix_lock
(
	Delta_Matrix C
);

void Delta_Matrix_unlock
(
	Delta_Matrix C
);

void Delta_Matrix_setDirty
(
	Delta_Matrix C
);

// print and check a GrB_Matrix
GrB_Info Delta_Matrix_fprint
(
    Delta_Matrix A,  // object to print and check
    int pr,          // print level (GxB_Print_Level)
    FILE *f          // file for output
) ;

void Delta_Matrix_free
(
	Delta_Matrix *C
);

