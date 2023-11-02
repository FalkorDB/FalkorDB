//------------------------------------------------------------------------------
// GB_Pending_free: free a list of pending tuples
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_Pending.h"

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
        GB_FREE (&(Pending->i), Pending->i_size) ;
        GB_FREE (&(Pending->j), Pending->j_size) ;
        GB_FREE (&(Pending->x), Pending->x_size) ;
        GB_FREE (&(Pending), Pending->header_size) ;
    }

    (*PHandle) = NULL ;
}

