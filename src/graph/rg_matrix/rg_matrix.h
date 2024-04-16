/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "RG.h"
#include "GraphBLAS.h"

#include <pthread.h>

// forward declaration of RG_Matrix type
typedef struct _RG_Matrix _RG_Matrix;
typedef _RG_Matrix *RG_Matrix;


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

GrB_Info RG_Matrix_new
(
	RG_Matrix *A,            // handle of matrix to create
	GrB_Type type,           // type of matrix to create
	GrB_Index nrows,         // matrix dimension is nrows-by-ncols
	GrB_Index ncols,
	bool transpose           // if true, create a transpose of the matrix
);

// returns transposed matrix of C
RG_Matrix RG_Matrix_getTranspose
(
	const RG_Matrix C
);

bool RG_Matrix_isDirty
(
	const RG_Matrix C
);

GrB_Matrix RG_Matrix_M
(
	const RG_Matrix C
);

GrB_Matrix RG_Matrix_DP
(
	const RG_Matrix C
);

GrB_Matrix RG_Matrix_DM
(
	const RG_Matrix C
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

GrB_Info RG_Matrix_extractElement_BOOL     // x = A(i,j)
(
	bool *x,                               // extracted scalar
	const RG_Matrix A,                     // matrix to extract a scalar from
	GrB_Index i,                           // row index
	GrB_Index j                            // column index
) ;

GrB_Info RG_Matrix_extract_row
(
	const RG_Matrix A,                      // matrix to extract a vector from
	GrB_Vector v,                           // vector to extract
	GrB_Index i                             // row index
) ;

// remove entry at position C[i,j]
GrB_Info RG_Matrix_removeElement_BOOL
(
	RG_Matrix C,                    // matrix to remove entry from
	GrB_Index i,                    // row index
	GrB_Index j                     // column index
);

GrB_Info RG_Matrix_removeElements
(
	RG_Matrix C,                    // matrix to remove entry from
	GrB_Matrix m                    // elements to remove
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

void RG_Matrix_synchronize
(
	RG_Matrix C,
	GrB_Index nrows,
	GrB_Index ncols
);

void RG_Matrix_free
(
	RG_Matrix *C
);

