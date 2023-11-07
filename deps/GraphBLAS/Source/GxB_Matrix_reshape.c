//------------------------------------------------------------------------------
// GxB_Matrix_reshape:  reshape a matrix in place
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GxB_Matrix_reshape changes the dimensions of a matrix, reshaping the entries
// by row or by column.

// For example, if C is 3-by-4 on input, and is reshaped by column to have
// dimensions 2-by-6:

//      C on input      C on output (by_col true)
//      00 01 02 03     00 20 11 02 22 13
//      10 11 12 13     10 01 21 12 03 23
//      20 21 22 23

// If the same C on input is reshaped by row to dimesions 2-by-6:

//      C on input      C on output (by_col false)
//      00 01 02 03     00 01 02 03 10 11
//      10 11 12 13     12 13 20 21 22 23
//      20 21 22 23

// If the input matrix is nrows-by-ncols, and the size of the reshaped matrix
// is nrows_new-by-ncols_new, then nrows*ncols must equal nrows_new*ncols_new.
// The format of the input matrix (by row or by column) is unchanged; this
// format need not match the by_col input parameter.

#include "GB.h"
#include "GB_reshape.h"

GrB_Info GxB_Matrix_reshape     // reshape a GrB_Matrix in place
(
    // input/output:
    GrB_Matrix C,               // input/output matrix, reshaped in place
    // input:
    bool by_col,                // true if reshape by column, false if by row
    GrB_Index nrows_new,        // new number of rows of C
    GrB_Index ncols_new,        // new number of columns of C
    const GrB_Descriptor desc   // to control # of threads used
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_reshape (C, nrows_new, ncols_new, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_reshape") ;
    GB_RETURN_IF_NULL_OR_FAULTY (C) ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;

    //--------------------------------------------------------------------------
    // reshape the matrix
    //--------------------------------------------------------------------------

    info = GB_reshape (NULL, C, by_col, nrows_new, ncols_new, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

