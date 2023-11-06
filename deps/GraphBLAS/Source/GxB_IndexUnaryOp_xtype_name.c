//------------------------------------------------------------------------------
// GxB_IndexUnaryOp_xtype_name: return the type_name of x for z=f(x,i,j,y)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_IndexUnaryOp_xtype_name    // return the name of the type of x
(
    char *type_name,        // name of the type (char array of size at least
                            // GxB_MAX_NAME_LEN, owned by the user application).
    const GrB_IndexUnaryOp op
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_IndexUnaryOp_xtype_name (type_name, op)") ;
    GB_RETURN_IF_NULL (type_name) ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    ASSERT_INDEXUNARYOP_OK (op, "op for xtype_name", GB0) ;

    //--------------------------------------------------------------------------
    // get the type_name
    //--------------------------------------------------------------------------

    memset (type_name, 0, GxB_MAX_NAME_LEN) ;
    if (op->xtype != NULL)
    { 
        memcpy (type_name, op->xtype->name, GxB_MAX_NAME_LEN) ;
    }
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

