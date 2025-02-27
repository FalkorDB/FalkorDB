//------------------------------------------------------------------------------
// GrB_Scalar_wait: wait for a scalar to complete
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Finishes all work on a scalar, followed by an OpenMP flush.

#include "GB.h"

#define GB_FREE_ALL ;

GrB_Info GrB_Scalar_wait    // finish all work on a scalar
(
    GrB_Scalar s,
    int waitmode
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (s) ;
    GB_WHERE1 (s, "GrB_Scalar_wait (s, waitmode)") ;

    //--------------------------------------------------------------------------
    // finish all pending work on the scalar
    //--------------------------------------------------------------------------

    if (waitmode != GrB_COMPLETE && GB_ANY_PENDING_WORK (s))
    { 
        GB_BURBLE_START ("GrB_Scalar_wait") ;
        GB_OK (GB_wait ((GrB_Matrix) s, "scalar", Werk)) ;
        GB_BURBLE_END ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Scalar_wait: wait for a scalar to complete (historical)
//------------------------------------------------------------------------------

GrB_Info GxB_Scalar_wait    // finish all work on a scalar
(
    GrB_Scalar *s
)
{
    return (GrB_Scalar_wait (*s, GrB_MATERIALIZE)) ;
}

