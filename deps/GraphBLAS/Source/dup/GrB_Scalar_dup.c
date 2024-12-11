//------------------------------------------------------------------------------
// GrB_Scalar_dup: make a deep copy of a sparse GrB_Scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// s = t, making a deep copy

#include "GB.h"

GrB_Info GrB_Scalar_dup     // make an exact copy of a GrB_Scalar
(
    GrB_Scalar *s,          // handle of output GrB_Scalar to create
    const GrB_Scalar t      // input GrB_Scalar to copy
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Scalar_dup (&s, t)") ;
    GB_RETURN_IF_NULL (s) ;
    GB_RETURN_IF_NULL_OR_FAULTY (t) ;
    ASSERT (GB_SCALAR_OK (t)) ;

    //--------------------------------------------------------------------------
    // duplicate the GrB_Scalar
    //--------------------------------------------------------------------------

    return (GB_dup ((GrB_Matrix *) s, (GrB_Matrix) t, Werk)) ;
}

//------------------------------------------------------------------------------
// GxB_Scalar_dup: make a deep copy of a sparse GrB_Scalar (historical)
//------------------------------------------------------------------------------

GrB_Info GxB_Scalar_dup     // make an exact copy of a GrB_Scalar
(
    GrB_Scalar *s,          // handle of output GrB_Scalar to create
    const GrB_Scalar t      // input GrB_Scalar to copy
)
{
    return (GrB_Scalar_dup (s, t)) ;
}

