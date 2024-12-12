//------------------------------------------------------------------------------
// GrB_BinaryOp_wait: wait for a user-defined GrB_BinaryOp to complete
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// In SuiteSparse:GraphBLAS, a user-defined GrB_BinaryOp has no pending
// operations to wait for.  All this method does is verify that the op is
// properly initialized, and then it does an OpenMP flush.

#include "GB.h"

GrB_Info GrB_BinaryOp_wait   // no work, just check if the GrB_BinaryOp is valid
(
    GrB_BinaryOp op,
    GrB_WaitMode waitmode
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_BinaryOp_wait (op, waitmode)") ;
    if (op == GxB_IGNORE_DUP) return (GrB_SUCCESS) ;    // nothing to do
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

