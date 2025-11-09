//------------------------------------------------------------------------------
// GB_block: apply all pending computations if blocking mode enabled
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "pending/GB_Pending.h"

#define GB_FREE_ALL ;

GrB_Info GB_block   // apply all pending computations if blocking mode enabled
(
    GrB_Matrix A,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (A != NULL) ;

    //--------------------------------------------------------------------------
    // wait if mode is blocking, or if too many pending tuples
    //--------------------------------------------------------------------------

    if (!(GB_ANY_PENDING_WORK (A) || GB_hyper_hash_need (A)))
    { 
        // no pending work, so no need to block
        return (GrB_SUCCESS) ;
    }

    double npending = (double) GB_Pending_n (A) ;
    double anzmax = ((double) A->vlen) * ((double) A->vdim) ;
    bool many_pending = (npending >= anzmax) ;
    int mode = GB_Global_mode_get ( ) ;
    bool blocking = (mode == GrB_BLOCKING || mode == GxB_BLOCKING_GPU) ;

    if (many_pending || blocking)
    { 
        // delete any lingering zombies, assemble any pending tuples,
        // sort the vectors, and construct the A->Y hyper_hash
        GB_OK (GB_wait (A, "matrix", Werk)) ;
        GB_OK (GB_hyper_hash_build (A, Werk)) ;
    }

    return (GrB_SUCCESS) ;
}

