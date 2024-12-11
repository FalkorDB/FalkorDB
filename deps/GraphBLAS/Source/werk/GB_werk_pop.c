//------------------------------------------------------------------------------
// GB_werk_pop:  free werkspace from the Werk stack
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

// If the werkspace was allocated from the Werk stack, it must be at the top of
// the stack to free it properly.  Freeing a werkspace in the middle of the
// Werk stack also frees everything above it.  This is not a problem if that
// space is also being freed, but the assertion below ensures that the freeing
// werkspace from the Werk stack is done in LIFO order, like a stack.

GB_CALLBACK_WERK_POP_PROTO (GB_werk_pop)
{
    ASSERT (size_allocated != NULL) ;

    if (p == NULL)
    { 
        // nothing to do
    }
    else if (on_stack)
    { 
        // werkspace was allocated from the Werk stack
        ASSERT ((*size_allocated) == GB_ROUND8 (nitems * size_of_item)) ;
        ASSERT (Werk != NULL) ;
        ASSERT ((*size_allocated) % 8 == 0) ;
        ASSERT (((GB_void *) p) + (*size_allocated) ==
                Werk->Stack + Werk->pwerk) ;
        Werk->pwerk = ((GB_void *) p) - Werk->Stack ;
        (*size_allocated) = 0 ;
    }
    else
    { 
        // werkspace was allocated from malloc
        GB_free_memory (&p, *size_allocated) ;
    }
    return (NULL) ;                 // return NULL to indicate p was freed
}

