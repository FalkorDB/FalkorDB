//------------------------------------------------------------------------------
// GxB_Semiring_multiply: return the multiply operator of a semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Semiring_multiply      // return multiply operator of a semiring
(
    GrB_BinaryOp *multiply,         // returns multiply operator of the semiring
    GrB_Semiring semiring           // semiring to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Semiring_multiply (&multiply, semiring)") ;
    GB_RETURN_IF_NULL (multiply) ;
    GB_RETURN_IF_NULL_OR_FAULTY (semiring) ;
    ASSERT_SEMIRING_OK (semiring, "semiring for mult", GB0) ;

    //--------------------------------------------------------------------------
    // return the ztype
    //--------------------------------------------------------------------------

    (*multiply) = semiring->multiply ;
    return (GrB_SUCCESS) ;
}

