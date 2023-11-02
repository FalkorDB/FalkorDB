//------------------------------------------------------------------------------
// GxB_UnaryOp_ztype: return the type of z for z=f(x)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// NOTE: this function is historical.  Use GxB_UnaryOp_ztype_name instead.

#include "GB.h"

GrB_Info GxB_UnaryOp_ztype          // return the type of z
(
    GrB_Type *ztype,                // return type of output z
    GrB_UnaryOp unaryop             // unary operator
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_UnaryOp_ztype (&ztype, unaryop)") ;
    GB_RETURN_IF_NULL (ztype) ;
    GB_RETURN_IF_NULL_OR_FAULTY (unaryop) ;
    ASSERT_UNARYOP_OK (unaryop, "unaryop for ztype", GB0) ;

    //--------------------------------------------------------------------------
    // return the ztype
    //--------------------------------------------------------------------------

    (*ztype) = unaryop->ztype ;
    return (GrB_SUCCESS) ;
}

