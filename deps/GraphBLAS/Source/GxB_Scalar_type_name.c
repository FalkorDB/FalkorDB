//------------------------------------------------------------------------------
// GxB_Scalar_type_name: return the name of the type of a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Scalar_type_name      // return the name of the type of a scalar
(
    char *type_name,        // name of the type (char array of size at least
                            // GxB_MAX_NAME_LEN, owned by the user application).
    const GrB_Scalar s
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Scalar_type_name (type_name, s)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (s) ;

    //--------------------------------------------------------------------------
    // get the type_name
    //--------------------------------------------------------------------------

    return (GB_matvec_type_name (type_name, (GrB_Matrix) s, Werk)) ;
}

