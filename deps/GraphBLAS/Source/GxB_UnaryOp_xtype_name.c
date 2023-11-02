//------------------------------------------------------------------------------
// GxB_UnaryOp_xtype_name: return the type_name of x for z=f(x)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_UnaryOp_xtype_name    // return the name of the type of x
(
    char *type_name,        // name of the type (char array of size at least
                            // GxB_MAX_NAME_LEN, owned by the user application).
    const GrB_UnaryOp unaryop
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_UnaryOp_xtype_name (type_name, op)") ;
    GB_RETURN_IF_NULL (type_name) ;
    GB_RETURN_IF_NULL_OR_FAULTY (unaryop) ;
    ASSERT_UNARYOP_OK (unaryop, "unaryop for xtype_name", GB0) ;

    //--------------------------------------------------------------------------
    // get the type_name
    //--------------------------------------------------------------------------

    memcpy (type_name, unaryop->xtype->name, GxB_MAX_NAME_LEN) ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

