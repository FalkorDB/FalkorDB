//------------------------------------------------------------------------------
// GxB_Matrix_reshapeDup:  reshape a matrix into another matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// See GxB_Matrix_reshape for a description of the output matrix C.

// If the input matrix A is nrows-by-ncols, and the size of the newly-created
// matrix C is nrows_new-by-ncols_new, then nrows*ncols must equal
// nrows_new*ncols_new.  The format of the input matrix A (by row or by column)
// determines the format of the output matrix C, which need not match the
// by_col input parameter.

// The matrix is allocated in arenas determined by the current Context.

#include "GB.h"
#include "reshape/GB_reshape.h"

GrB_Info GxB_Matrix_reshapeDup  // reshape a GrB_Matrix into another GrB_Matrix
(
    // output:
    GrB_Matrix *C,              // newly created output matrix, not in place
    // input:
    GrB_Matrix A,               // input matrix, not modified
    bool by_col,                // true if reshape by column, false if by row
    uint64_t nrows_new,         // number of rows of C
    uint64_t ncols_new,         // number of columns of C
    const GrB_Descriptor desc   // to control # of threads used
)
{ 
    int header_arena = GB_Context_header_arena ( ) ;
    int data_arena = GB_Context_data_arena ( ) ;
    return (GxB_Matrix_reshapeDup_arena (C, A, by_col, nrows_new, ncols_new,
        header_arena, data_arena, desc)) ;
}

