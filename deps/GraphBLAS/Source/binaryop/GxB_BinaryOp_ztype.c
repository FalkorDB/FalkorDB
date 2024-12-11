//------------------------------------------------------------------------------
// GxB_BinaryOp_ztype: return the type of z for z=f(x,y)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// NOTE: this function is historical.  Use GxB_BinaryOp_ztype_name instead.

#include "GB.h"

GrB_Info GxB_BinaryOp_ztype         // return the type of z
(
    GrB_Type *ztype,                // return type of output z
    GrB_BinaryOp binaryop           // binary operator to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_BinaryOp_ztype (&ztype, binaryop)") ;
    GB_RETURN_IF_NULL (ztype) ;
    GB_RETURN_IF_NULL_OR_FAULTY (binaryop) ;
    ASSERT_BINARYOP_OK (binaryop, "binaryop for ztype", GB0) ;

    //--------------------------------------------------------------------------
    // return the ztype
    //--------------------------------------------------------------------------

    (*ztype) = binaryop->ztype ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

