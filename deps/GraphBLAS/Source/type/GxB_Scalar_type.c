//------------------------------------------------------------------------------
// GxB_Scalar_type: return the type of a GrB_Scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Scalar_type    // get the type of a GrB_Scalar
(
    GrB_Type *type,         // returns the type of the GrB_Scalar
    const GrB_Scalar s      // GrB_Scalar to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (s) ;
    GB_WHERE_1 (s, "GxB_Scalar_type (&type, s)") ;

    //--------------------------------------------------------------------------
    // get the type
    //--------------------------------------------------------------------------

    return (GB_matvec_type (type, (GrB_Matrix) s, Werk)) ;
}

