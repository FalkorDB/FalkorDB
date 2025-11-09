//------------------------------------------------------------------------------
// GrB_Vector_wait: wait for a vector to complete
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Finishes all work on a vector, followed by an OpenMP flush.

#include "GB.h"

#define GB_FREE_ALL ;

GrB_Info GrB_Vector_wait    // finish all work on a vector
(
    GrB_Vector v,
    int waitmode
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (v) ;
    GB_WHERE1 (v, "GrB_Vector_wait (v, waitmode)") ;

    //--------------------------------------------------------------------------
    // finish all pending work on the vector
    //--------------------------------------------------------------------------

    if (waitmode != GrB_COMPLETE && GB_ANY_PENDING_WORK (v))
    { 
        GB_BURBLE_START ("GrB_Vector_wait") ;
        GB_OK (GB_wait ((GrB_Matrix) v, "vector", Werk)) ;
        GB_BURBLE_END ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

