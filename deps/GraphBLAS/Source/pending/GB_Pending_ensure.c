//------------------------------------------------------------------------------
// GB_Pending_ensure: ensure a list of pending tuples is large enough
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "pending/GB_Pending.h"

// create or reallocate a list of pending tuples

GB_CALLBACK_PENDING_ENSURE_PROTO (GB_Pending_ensure)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (PHandle != NULL) ;

    //--------------------------------------------------------------------------
    // ensure the list of pending tuples is large enough
    //--------------------------------------------------------------------------

    if ((*PHandle) == NULL)
    {
        return (GB_Pending_alloc (PHandle, iso, type, op, is_matrix, nnew)) ;
    }
    else
    {
        return (GB_Pending_realloc (PHandle, nnew, Werk)) ;
    }
}

