//------------------------------------------------------------------------------
// GxB_SelectOp_xtype: return the type of x for z=f(x,thunk)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// NOTE: this function is historical.  Use a GrB_IndexUnaryOp instead.

#include "GB.h"

GrB_Info GxB_SelectOp_xtype         // return the type of x or NULL if generic
(
    GrB_Type *xtype,                // return type of input x
    GxB_SelectOp selectop           // select operator
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_SelectOp_xtype (&xtype, selectop)") ;
    GB_RETURN_IF_NULL (xtype) ;
    GB_RETURN_IF_NULL_OR_FAULTY (selectop) ;
    ASSERT_SELECTOP_OK (selectop, "selectop for xtype", GB0) ;

    //--------------------------------------------------------------------------
    // return the xtype
    //--------------------------------------------------------------------------

    (*xtype) = selectop->xtype ;
    return (GrB_SUCCESS) ;
}

