//------------------------------------------------------------------------------
// GrB_Matrix_wait: wait for a matrix to complete
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Finishes all work on a matrix, followed by an OpenMP flush.

#include "GB.h"

#define GB_FREE_ALL ;

GrB_Info GrB_Matrix_wait    // finish all work on a matrix
(
    GrB_Matrix A,
    int waitmode
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (A) ;
    GB_WHERE1 (A, "GrB_Matrix_wait (A, waitmode)") ;

    //--------------------------------------------------------------------------
    // finish all pending work on the matrix, including creating A->Y
    //--------------------------------------------------------------------------

    if (waitmode != GrB_COMPLETE &&
        (GB_ANY_PENDING_WORK (A) || GB_hyper_hash_need (A)))
    { 
        GB_BURBLE_START ("GrB_Matrix_wait") ;
        GB_OK (GB_wait (A, "matrix", Werk)) ;
        GB_OK (GB_hyper_hash_build (A, Werk)) ;
        GB_BURBLE_END ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

