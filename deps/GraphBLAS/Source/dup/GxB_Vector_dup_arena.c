//------------------------------------------------------------------------------
// GxB_Vector_dup_arena: make a deep copy of a sparse vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// w = u, making a deep copy

// The arenas for w are given by input parameters (checked by GB_dup)

#include "GB.h"

GrB_Info GxB_Vector_dup_arena // make an exact copy of a vector
(
    GrB_Vector *w,          // handle of output vector to create
    const GrB_Vector u,     // input vector to copy
    const int header_arena,
    const int data_arena
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (w) ;
    GB_RETURN_IF_NULL (u) ;
    GB_WHERE_1 (u, "GxB_Vector_dup_arena (&w, u, header_arena, data_arena)") ;
    GB_BURBLE_START ("GrB_Vector_dup") ;

    ASSERT (GB_VECTOR_OK (u)) ;

    //--------------------------------------------------------------------------
    // duplicate the vector
    //--------------------------------------------------------------------------

    info = GB_dup ((GrB_Matrix *) w, (GrB_Matrix) u,
        header_arena, data_arena, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

