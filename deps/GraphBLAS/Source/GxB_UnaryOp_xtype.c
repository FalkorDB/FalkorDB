//------------------------------------------------------------------------------
// GxB_UnaryOp_xtype: return the type of x for z=f(x)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// NOTE: this function is historical.  Use GxB_UnaryOp_xtype_name instead.

#include "GB.h"

GrB_Info GxB_UnaryOp_xtype          // return the type of x
(
    GrB_Type *xtype,                // return type of input x
    GrB_UnaryOp unaryop             // unary operator
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_UnaryOp_xtype (&xtype, unaryop)") ;
    GB_RETURN_IF_NULL (xtype) ;
    GB_RETURN_IF_NULL_OR_FAULTY (unaryop) ;
    ASSERT_UNARYOP_OK (unaryop, "unaryop for xtype", GB0) ;

    //--------------------------------------------------------------------------
    // return the xtype
    //--------------------------------------------------------------------------

    (*xtype) = unaryop->xtype ;
    return (GrB_SUCCESS) ;
}

