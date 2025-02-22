//------------------------------------------------------------------------------
// GrB_Matrix_resize: change the size of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "resize/GB_resize.h"

GrB_Info GrB_Matrix_resize      // change the size of a matrix
(
    GrB_Matrix C,               // matrix to modify
    uint64_t nrows_new,         // new number of rows in matrix
    uint64_t ncols_new          // new number of columns in matrix
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (C) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (C) ;
    GB_WHERE1 (C, "GrB_Matrix_resize (C, nrows_new, ncols_new)") ;
    GB_BURBLE_START ("GrB_Matrix_resize") ;

    //--------------------------------------------------------------------------
    // resize the matrix
    //--------------------------------------------------------------------------

    info = GB_resize (C, nrows_new, ncols_new, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GxB_Matrix_resize: historical
//------------------------------------------------------------------------------

// This function now appears in the C API Specification as GrB_Matrix_resize.
// The new name is preferred.  The old name will be kept for historical
// compatibility.

GrB_Info GxB_Matrix_resize      // change the size of a matrix
(
    GrB_Matrix A,               // matrix to modify
    uint64_t nrows_new,         // new number of rows in matrix
    uint64_t ncols_new          // new number of columns in matrix
)
{ 
    return (GrB_Matrix_resize (A, nrows_new, ncols_new)) ;
}

