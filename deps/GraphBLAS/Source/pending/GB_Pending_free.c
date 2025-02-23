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
        GB_FREE_MEMORY (&(Pending->i), Pending->i_size) ;
        GB_FREE_MEMORY (&(Pending->j), Pending->j_size) ;
        GB_FREE_MEMORY (&(Pending->x), Pending->x_size) ;
        GB_FREE_MEMORY (&(Pending), Pending->header_size) ;
    }

    (*PHandle) = NULL ;
}

