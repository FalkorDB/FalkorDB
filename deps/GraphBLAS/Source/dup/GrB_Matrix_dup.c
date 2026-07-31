//------------------------------------------------------------------------------
// GrB_Matrix_dup: make a deep copy of a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = A, making a deep copy

// The matrix is allocated in arenas determined by the current Context.

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

    GB_RETURN_IF_NULL (C) ;
    GB_RETURN_IF_NULL (A) ;
    GB_WHERE_1 (A, "GrB_Matrix_dup (&C, A)") ;
    GB_BURBLE_START ("GrB_Matrix_dup") ;

    int header_arena = GB_Context_header_arena ( ) ;
    int data_arena = GB_Context_data_arena ( ) ;

    //--------------------------------------------------------------------------
    // duplicate the matrix
    //--------------------------------------------------------------------------

    info = GB_dup (C, A, header_arena, data_arena, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

