/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "RG.h"
#include "GraphBLAS.h"

// forward declaration of Delta_Matrix type
typedef struct _Delta_Matrix _Delta_Matrix;
typedef _Delta_Matrix *Delta_Matrix;


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

GrB_Info Delta_Matrix_new
(
	Delta_Matrix *A,         // handle of matrix to create
	GrB_Type type,           // type of matrix to create
	GrB_Index nrows,         // matrix dimension is nrows-by-ncols
	GrB_Index ncols,
	bool transpose           // if true, create a transpose of the matrix
);

GrB_Matrix Delta_Matrix_M
(
	const Delta_Matrix C
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

GrB_Info Delta_Matrix_nvals    // get the number of entries in a matrix
(
	GrB_Index *nvals,       // matrix has nvals entries
	const Delta_Matrix A    // matrix to query
);

GrB_Info Delta_Matrix_setElement_BOOL      // C (i,j) = x
(
	Delta_Matrix C,                     // matrix to modify
	GrB_Index i,                        // row index
	GrB_Index j                         // column index
);

GrB_Info Delta_Matrix_extractElement_BOOL     // x = A(i,j)
(
	bool *x,                               // extracted scalar
	const Delta_Matrix A,                  // matrix to extract a scalar from
	GrB_Index i,                           // row index
	GrB_Index j                            // column index
) ;

GrB_Info Delta_mxm                     // C = A * B
(
	Delta_Matrix C,                    // input/output matrix for results
	const GrB_Semiring semiring,       // defines '+' and '*' for A*B
	const Delta_Matrix A,              // first input:  matrix A
	const Delta_Matrix B               // second input: matrix B
);

GrB_Info Delta_eWiseAdd                // C = A + B
(
    Delta_Matrix C,                    // input/output matrix for results
    const GrB_Semiring semiring,       // defines '+' for T=A+B
    const Delta_Matrix A,              // first input:  matrix A
    const Delta_Matrix B               // second input: matrix B
);

GrB_Info Delta_Matrix_clear    // clear a matrix of all entries;
(                           // type and dimensions remain unchanged
    Delta_Matrix A          // matrix to clear
);

GrB_Info Delta_Matrix_copy     // copy matrix A to matrix C
(
	Delta_Matrix C,            // output matrix
	const Delta_Matrix A       // input matrix
);

// get matrix C without writing to internal matrix
GrB_Info Delta_Matrix_export
(
	GrB_Matrix *A,
	Delta_Matrix C
);

GrB_Info Delta_Matrix_wait
(
	Delta_Matrix C,
	bool force_sync
);

void Delta_Matrix_free
(
	Delta_Matrix *C
);

