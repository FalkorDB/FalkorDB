//------------------------------------------------------------------------------
// GrB_Matrix_new: create a new matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The new matrix is nrows-by-ncols, with no entries in it.

#include "GB.h"

GrB_Info GrB_Matrix_new     // create a new matrix with no entries
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create
    GrB_Index nrows,        // matrix dimension is nrows-by-ncols
    GrB_Index ncols
)
{ 
    GB_WHERE1 ("GrB_Matrix_new (&A, type, nrows, ncols)") ;
    return (GB_Matrix_new (A, type, nrows, ncols)) ;
}

