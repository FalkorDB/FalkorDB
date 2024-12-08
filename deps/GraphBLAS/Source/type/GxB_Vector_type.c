//------------------------------------------------------------------------------
// GxB_Vector_type: return the type of a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// NOTE: this function is historical.  Use GxB_Vector_type_name instead.

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

    GB_WHERE1 ("GxB_Vector_type (&type, v)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (v) ;

    //--------------------------------------------------------------------------
    // get the type
    //--------------------------------------------------------------------------

    return (GB_matvec_type (type, (GrB_Matrix) v, Werk)) ;
}

