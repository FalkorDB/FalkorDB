//------------------------------------------------------------------------------
// GxB_Vector_type_name: return the name of the type of a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Vector_type_name      // return the name of the type of a vector
(
    char *type_name,        // name of the type (char array of size at least
                            // GxB_MAX_NAME_LEN, owned by the user application).
    const GrB_Vector v
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (v) ;
    GB_WHERE_1 (v, "GxB_Vector_type_name (type_name, v)") ;

    //--------------------------------------------------------------------------
    // get the type_name
    //--------------------------------------------------------------------------

    return (GB_matvec_type_name (type_name, (GrB_Matrix) v, Werk)) ;
}

