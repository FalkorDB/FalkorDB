//------------------------------------------------------------------------------
// GB_werk_push: allocate werkspace from the Werk stack or malloc
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

// The werkspace is allocated from the Werk static if it small enough and space
// is available.  Otherwise it is allocated by malloc.

#ifdef comments_only
void *GB_werk_push    // return pointer to newly allocated space
(
    // output
    uint64_t *p_mem,        // memsize and memlane of allocated space
    bool *on_stack,         // true if werkspace is from Werk stack
    // input
    uint64_t nitems,        // # of items to allocate
    uint64_t size_of_item,  // size of each item
    GB_Werk Werk
) ;
#endif

GB_CALLBACK_WERK_PUSH_PROTO (GB_werk_push)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (on_stack != NULL) ;
    ASSERT (p_mem != NULL) ;

    int memlane = 0 ;   // FIXME memlane from Context
    uint64_t mem = GB_mem (memlane, 0) ;

    //--------------------------------------------------------------------------
    // determine where to allocate the werkspace
    //--------------------------------------------------------------------------

    uint64_t memsize ;
    if (Werk == NULL || nitems > GB_WERK_SIZE || size_of_item > GB_WERK_SIZE
        #ifdef GBCOVER
        // Werk stack can be disabled for test coverage
        || (GB_Global_hack_get (1) != 0)
        #endif
    )
    { 
        // no context, or werkspace is too large to allocate from the Werk stack
        (*on_stack) = false ;
    }
    else
    { 
        // try to allocate from the Werk stack
        memsize = GB_ROUND8 (nitems * size_of_item) ;
        ASSERT (memsize % 8 == 0) ; // memsize is rounded up to a multiple of 8
        uint64_t freespace = GB_WERK_SIZE - Werk->pwerk ;
        ASSERT (freespace % 8 == 0) ;   // thus freespace is also multiple of 8
        (*on_stack) = (memsize <= freespace) ;
    }

    //--------------------------------------------------------------------------
    // allocate the werkspace
    //--------------------------------------------------------------------------

    if (*on_stack)
    { 
        // allocate the werkspace from the Werk stack
        GB_void *p = Werk->Stack + Werk->pwerk ;
        Werk->pwerk += (int) memsize ;
        (*p_mem) = GB_mem (0, memsize) ;
        return ((void *) p) ;
    }
    else
    { 
        // allocate the werkspace from malloc
        (*p_mem) = mem ;
        void *p = GB_MALLOC_MEMORY (nitems, size_of_item, p_mem) ;
        return (p) ;
    }
}

