//------------------------------------------------------------------------------
// GB_Pending_ensure: ensure a list of pending tuples is large enough
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "pending/GB_Pending.h"

// create or reallocate a list of pending tuples

GB_CALLBACK_PENDING_ENSURE_PROTO (GB_Pending_ensure)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (C != NULL) ;

    //--------------------------------------------------------------------------
    // ensure the list of pending tuples is large enough
    //--------------------------------------------------------------------------

    if (C->Pending == NULL)
    {
        return (GB_Pending_alloc (C, iso, type, op, nnew)) ;
    }
    else
    {
        return (GB_Pending_realloc (C, nnew, Werk)) ;
    }
}

