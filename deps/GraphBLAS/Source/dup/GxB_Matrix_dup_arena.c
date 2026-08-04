//------------------------------------------------------------------------------
// GxB_Matrix_dup_arena: make a deep copy of a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = A, making a deep copy

// The arenas for C are given by input parameters.

#include "GB.h"

GrB_Info GxB_Matrix_dup_arena     // make an exact copy of a matrix
(
    GrB_Matrix *C,          // handle of output matrix to create
    const GrB_Matrix A,     // input matrix to copy
    const int header_arena,
    const int data_arena
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (C) ;
    GB_RETURN_IF_NULL (A) ;
    GB_WHERE_1 (A, "GxB_Matrix_dup_arena (&C, A, header_arena, data_arena)") ;
    GB_BURBLE_START ("GrB_Matrix_dup") ;

    //--------------------------------------------------------------------------
    // duplicate the matrix
    //--------------------------------------------------------------------------

    info = GB_dup (C, A, header_arena, data_arena, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

