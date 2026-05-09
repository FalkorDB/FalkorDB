//------------------------------------------------------------------------------
// GB_werk_pop:  free werkspace from the Werk stack
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

// If the werkspace was allocated from the Werk stack, it must be at the top of
// the stack to free it properly.  Freeing a werkspace in the middle of the
// Werk stack also frees everything above it.  This is not a problem if that
// space is also being freed, but the assertion below ensures that the freeing
// werkspace from the Werk stack is done in LIFO order, like a stack.

#ifdef comments_only
void *GB_werk_pop     // free the top block of werkspace memory
(
    // input/output
    void *p,                    // werkspace to free
    uint64_t *mem,              // memsize and memlane of p
    // input
    bool on_stack,              // true if werkspace is from Werk stack
    uint64_t nitems,            // # of items to allocate
    uint64_t size_of_item,      // size of each item
    GB_Werk Werk
) ;
#endif

GB_CALLBACK_WERK_POP_PROTO (GB_werk_pop)
{
    ASSERT (mem != NULL) ;

    if (p == NULL)
    { 
        // nothing to do
    }
    else if (on_stack)
    { 
        // werkspace was allocated from the Werk stack
        ASSERT (GB_memsize (*mem) == GB_ROUND8 (nitems * size_of_item)) ;
        ASSERT (Werk != NULL) ;
        ASSERT (GB_memsize (*mem) % 8 == 0) ;
        ASSERT (((GB_void *) p) + GB_memsize (*mem) ==
                Werk->Stack + Werk->pwerk) ;
        Werk->pwerk = ((GB_void *) p) - Werk->Stack ;
        (*mem) = 0 ;
    }
    else
    { 
        // werkspace was allocated from malloc
        GB_free_memory (&p, *mem) ;
        (*mem) = 0 ;
    }
    return (NULL) ;                 // return NULL to indicate p was freed
}

