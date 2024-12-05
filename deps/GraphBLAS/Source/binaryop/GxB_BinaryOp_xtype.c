//------------------------------------------------------------------------------
// GxB_BinaryOp_xtype: return the type of x for z=f(x,y)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// NOTE: this function is historical.  Use GxB_BinaryOp_xtype_name instead.

#include "GB.h"

GrB_Info GxB_BinaryOp_xtype         // type of x
(
    GrB_Type *xtype,                // return type of input x
    GrB_BinaryOp binaryop           // binary operator to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_BinaryOp_xtype (&xtype, binaryop)") ;
    GB_RETURN_IF_NULL (xtype) ;
    GB_RETURN_IF_NULL_OR_FAULTY (binaryop) ;
    ASSERT_BINARYOP_OK (binaryop, "binaryop for xtype", GB0) ;

    //--------------------------------------------------------------------------
    // return the xtype
    //--------------------------------------------------------------------------

    (*xtype) = binaryop->xtype ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

