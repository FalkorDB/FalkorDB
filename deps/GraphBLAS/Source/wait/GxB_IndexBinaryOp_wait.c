//------------------------------------------------------------------------------
// GxB_IndexBinaryOp_wait: wait for a user-defined GxB_IndexBinaryOp to complete
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// In SuiteSparse:GraphBLAS, a user-defined GxB_IndexBinaryOp has no pending
// operations to wait for.  All this method does is verify that the op is
// properly initialized, and then it does an OpenMP flush.

#include "GB.h"

GrB_Info GxB_IndexBinaryOp_wait   // no work, just check if valid
(
    GxB_IndexBinaryOp op,
    int waitmode
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

