//------------------------------------------------------------------------------
// GrB_Scalar_clear: clears the content of a GrB_Scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// See GrB_Matrix_clear for details.

#include "GB.h"

GrB_Info GrB_Scalar_clear   // clear a GrB_Scalar of its entry
(                           // type and dimension remain unchanged
    GrB_Scalar s            // GrB_Scalar to clear
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 (s, "GrB_Scalar_clear (s)") ;
    GB_RETURN_IF_NULL (s) ;

    ASSERT (GB_SCALAR_OK (s)) ;

    //--------------------------------------------------------------------------
    // clear the GrB_Scalar
    //--------------------------------------------------------------------------

    return (GB_clear ((GrB_Matrix) s, Werk)) ;
}

//------------------------------------------------------------------------------
// GxB_Scalar_clear: clears the content of a GrB_Scalar (historical)
//------------------------------------------------------------------------------

GrB_Info GxB_Scalar_clear   // clear a GrB_Scalar of its entry
(                           // type and dimension remain unchanged
    GrB_Scalar s            // GrB_Scalar to clear
)
{
    return (GrB_Scalar_clear (s)) ;
}

