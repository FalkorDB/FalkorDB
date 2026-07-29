//------------------------------------------------------------------------------
// GxB_Scalar_dup_arena: make a deep copy of a sparse GrB_Scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// s = t, making a deep copy

// The arenas for s are given by input parameters.

#include "GB.h"

GrB_Info GxB_Scalar_dup_arena // make an exact copy of a GrB_Scalar
(
    GrB_Scalar *s,          // handle of output GrB_Scalar to create
    const GrB_Scalar t,     // input GrB_Scalar to copy
    const int header_arena,
    const int data_arena
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (s) ;
    GB_WHERE_1 (t, "GxB_Scalar_dup_arena (&s, t, header_arena, data_arena)") ;

    ASSERT (GB_SCALAR_OK (t)) ;

    //--------------------------------------------------------------------------
    // duplicate the GrB_Scalar
    //--------------------------------------------------------------------------

    return (GB_dup ((GrB_Matrix *) s, (GrB_Matrix) t,
        header_arena, data_arena, Werk)) ;
}

