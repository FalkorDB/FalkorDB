//------------------------------------------------------------------------------
// GB_Pending_free: free a list of pending tuples
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "pending/GB_Pending.h"

void GB_Pending_free        // free a list of pending tuples
(
    GB_Pending *PHandle
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (PHandle != NULL) ;

    //--------------------------------------------------------------------------
    // free all pending tuples
    //--------------------------------------------------------------------------

    GB_Pending Pending = (*PHandle) ;
    if (Pending != NULL)
    { 
        GB_FREE_MEMORY (&(Pending->i), Pending->i_mem) ;
        GB_FREE_MEMORY (&(Pending->j), Pending->j_mem) ;
        GB_FREE_MEMORY (&(Pending->x), Pending->x_mem) ;
        uint64_t header_mem = Pending->header_mem ;
        Pending->header_mem = 0 ;   // header is freed
        GB_FREE_MEMORY (PHandle, header_mem) ;
    }

    (*PHandle) = NULL ;
}

