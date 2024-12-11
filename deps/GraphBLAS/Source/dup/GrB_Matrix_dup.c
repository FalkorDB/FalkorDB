//------------------------------------------------------------------------------
// GrB_Matrix_dup: make a deep copy of a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = A, making a deep copy

#include "GB.h"

GrB_Info GrB_Matrix_dup     // make an exact copy of a matrix
(
    GrB_Matrix *C,          // handle of output matrix to create
    const GrB_Matrix A      // input matrix to copy
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Matrix_dup (&C, A)") ;
    GB_BURBLE_START ("GrB_Matrix_dup") ;
    GB_RETURN_IF_NULL (C) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;

    //--------------------------------------------------------------------------
    // duplicate the matrix
    //--------------------------------------------------------------------------

    GrB_Info info = GB_dup (C, A, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

