//------------------------------------------------------------------------------
// GrB_Vector_dup: make a deep copy of a sparse vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// w = u, making a deep copy

#include "GB.h"

GrB_Info GrB_Vector_dup     // make an exact copy of a vector
(
    GrB_Vector *w,          // handle of output vector to create
    const GrB_Vector u      // input vector to copy
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (w) ;
    GB_RETURN_IF_NULL (u) ;
    GB_WHERE_1 (u, "GrB_Vector_dup (&w, u)") ;
    GB_BURBLE_START ("GrB_Vector_dup") ;

    ASSERT (GB_VECTOR_OK (u)) ;

    //--------------------------------------------------------------------------
    // duplicate the vector
    //--------------------------------------------------------------------------

    info = GB_dup ((GrB_Matrix *) w, (GrB_Matrix) u, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

