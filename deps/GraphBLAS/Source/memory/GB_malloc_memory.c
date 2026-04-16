//------------------------------------------------------------------------------
// GB_malloc_memory: wrapper for malloc
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A wrapper for malloc.  Space is not initialized

#include "GB.h"

//------------------------------------------------------------------------------
// GB_malloc_helper:  malloc to allocate an non-initialized block
//------------------------------------------------------------------------------

static inline void *GB_malloc_helper
(
    // input/output:
    uint64_t *memsize,      // on input: # of bytes requested
                            // on output: # of bytes actually allocated
    // input
    int memlane
)
{
    void *p = NULL ;

    // make sure the block is at least 8 bytes in size
    (*memsize) = GB_IMAX (*memsize, 8) ;

    p = GB_Global_malloc_function (*memsize, memlane) ;

    #ifdef GB_MEMDUMP
    GBMDUMP ("malloc  %p %8ld: lane:%d ", p, *memsize, memlane) ;
    GB_Global_memtable_dump ( ) ;
    #endif

    return (p) ;
}

//------------------------------------------------------------------------------
// GB_malloc_memory
//------------------------------------------------------------------------------

#if 0
void *GB_malloc_memory      // pointer to allocated block of memory
(
    uint64_t nitems,        // number of items to allocate
    uint64_t size_of_item,  // sizeof each item
    // input/output
    uint64_t *mem           // # of bytes actually allocated, and memlane
)
#endif

GB_CALLBACK_CALLOC_MEMORY_PROTO (GB_malloc_memory)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (mem != NULL) ;

    void *p ;
    uint64_t memsize = 0 ;
    int memlane = GB_memlane (*mem) ;

    // make sure at least one item is allocated
    nitems = GB_IMAX (1, nitems) ;

    // make sure at least one byte is allocated
    size_of_item = GB_IMAX (1, size_of_item) ;

    bool ok = GB_uint64_multiply (&memsize, nitems, size_of_item) ;
    if (!ok || nitems > GB_NMAX || size_of_item > GB_NMAX)
    { 
        // overflow
        (*mem) = GB_mem (memlane, 0) ;
        return (NULL) ;
    }

    //--------------------------------------------------------------------------
    // allocate the memory block
    //--------------------------------------------------------------------------

    if (GB_Global_malloc_tracking_get ( ))
    {

        //----------------------------------------------------------------------
        // for memory usage testing only
        //----------------------------------------------------------------------

        // brutal memory debug; pretend to fail if (count-- <= 0).
        bool pretend_to_fail = false ;
        if (GB_Global_malloc_debug_get ( ))
        {
            pretend_to_fail = GB_Global_malloc_debug_count_decrement ( ) ;
        }

        // allocate the memory
        if (pretend_to_fail)
        { 
            p = NULL ;
        }
        else
        { 
            p = GB_malloc_helper (&memsize, memlane) ;
        }

    }
    else
    { 

        //----------------------------------------------------------------------
        // normal use, in production
        //----------------------------------------------------------------------

        p = GB_malloc_helper (&memsize, memlane) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    memsize = (p == NULL) ? 0 : memsize ;
    if (p != NULL)
    {
        MEMTABLE_ASSERT (memsize == GB_Global_memtable_memsize (p)) ;
        MEMTABLE_ASSERT (memlane == GB_Global_memtable_memlane (p)) ;
    }
    (*mem) = GB_mem (memlane, memsize) ;
    return (p) ;
}

