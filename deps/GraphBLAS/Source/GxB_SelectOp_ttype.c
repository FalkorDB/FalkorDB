//------------------------------------------------------------------------------
// GxB_SelectOp_ttype: return the type of thunk for z=f(x,thunk)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// NOTE: this function is historical.  Use a GrB_IndexUnaryOp instead.

#include "GB.h"

GrB_Info GxB_SelectOp_ttype         // return type of thunk or NULL if generic
(
    GrB_Type *ttype,                // return type of input thunk
    GxB_SelectOp selectop           // select operator
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_SelectOp_ttype (&ttype, selectop)") ;
    GB_RETURN_IF_NULL (ttype) ;
    GB_RETURN_IF_NULL_OR_FAULTY (selectop) ;
    ASSERT_SELECTOP_OK (selectop, "selectop for ttype", GB0) ;

    //--------------------------------------------------------------------------
    // return the ttype
    //--------------------------------------------------------------------------

    (*ttype) = selectop->ytype ;
    return (GrB_SUCCESS) ;
}

