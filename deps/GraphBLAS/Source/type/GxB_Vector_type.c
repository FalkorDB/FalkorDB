//------------------------------------------------------------------------------
// GxB_Vector_type: return the type of a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Vector_type    // get the type of a vector
(
    GrB_Type *type,         // returns the type of the vector
    const GrB_Vector v      // vector to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (v) ;
    GB_WHERE_1 (v, "GxB_Vector_type (&type, v)") ;

    //--------------------------------------------------------------------------
    // get the type
    //--------------------------------------------------------------------------

    return (GB_matvec_type (type, (GrB_Matrix) v, Werk)) ;
}

