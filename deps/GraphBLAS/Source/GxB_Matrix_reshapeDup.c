//------------------------------------------------------------------------------
// GxB_Matrix_reshapeDup:  reshape a matrix into another matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// See GxB_Matrix_reshape for a description of the output matrix c.

// If the input matrix A is nrows-by-ncols, and the size of the newly-created
// matrix C is nrows_new-by-ncols_new, then nrows*ncols must equal
// nrows_new*ncols_new.  The format of the input matrix A (by row or by column)
// determines the format of the output matrix C, which need not match the
// by_col input parameter.

#include "GB.h"
#include "GB_reshape.h"

GrB_Info GxB_Matrix_reshapeDup // reshape a GrB_Matrix into another GrB_Matrix
(
    // output:
    GrB_Matrix *C,              // newly created output matrix, not in place
    // input:
    GrB_Matrix A,               // input matrix, not modified
    bool by_col,                // true if reshape by column, false if by row
    GrB_Index nrows_new,        // number of rows of C
    GrB_Index ncols_new,        // number of columns of C
    const GrB_Descriptor desc   // to control # of threads used
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_reshapeDup (&C, A, nrows_new, ncols_new, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_reshapeDup") ;
    GB_RETURN_IF_NULL (C) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;

    //--------------------------------------------------------------------------
    // reshape the matrix
    //--------------------------------------------------------------------------

    info = GB_reshape (C, A, by_col, nrows_new, ncols_new, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

