//------------------------------------------------------------------------------
// GxB_Context_wait: wait for a GxB_Context to complete
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A GxB_Context has no pending operations to wait for.  All this method does
// is verify that the Context is properly initialized, and then it does an
// OpenMP flush.

#include "GB.h"

GrB_Info GxB_Context_wait      // no work, just check if GxB_Context is valid
(
    GxB_Context Context,
    GrB_WaitMode waitmode
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Context_wait (Context, waitmode)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (Context) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

