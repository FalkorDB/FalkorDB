//------------------------------------------------------------------------------
// GrB_Matrix_new: create a new matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The new matrix is nrows-by-ncols, with no entries in it.

// The matrix is allocated in the arenas determined by the current Context.

#include "GB.h"

GrB_Info GrB_Matrix_new     // create a new matrix with no entries
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create
    uint64_t nrows,         // matrix dimension is nrows-by-ncols
    uint64_t ncols
)
{ 
    int header_arena = GB_Context_header_arena ( ) ;
    int data_arena = GB_Context_data_arena ( ) ;
    return (GxB_Matrix_new_arena (A, type, nrows, ncols,
        header_arena, data_arena)) ;
}

