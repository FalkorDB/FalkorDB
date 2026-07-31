//------------------------------------------------------------------------------
// GB_calloc_memory: wrapper for calloc (actually uses malloc and memset)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A wrapper for calloc.  Space is set to zero on the CPU.
// CUDA will rely on its own method and will not call this method.

#include "GB.h"

#if 0
#ifdef GB_MEMDUMP
#include <execinfo.h>
#endif
#endif

//------------------------------------------------------------------------------
// GB_calloc_helper:  malloc/memset to allocate an initialized block
//------------------------------------------------------------------------------

static inline void *GB_calloc_helper
(
    // input/output:
    uint64_t *memsize,      // on input: # of bytes requested
                            // on output: # of bytes actually allocated
    // input
    int arena
)
{
    void *p = NULL ;

    // make sure the block is at least 8 bytes in size
    (*memsize) = GB_IMAX (*memsize, 8) ;

    #ifdef GB_MEMDUMP
    GBMDUMP ("\n------------- Starting calloc:\n") ;
    #if 0
    {
        // this only works for Linux
        int nptrs ;
        void *buffer [30] ;
        nptrs = backtrace (buffer, 30) ;
        backtrace_symbols_fd (buffer, nptrs, 0) ;
    }
    #endif
    #endif

    p = GB_Global_malloc_function (*memsize, arena) ;

    #ifdef GB_MEMDUMP
    GBMDUMP ("calloc  %p %8ld: arena:%d ", p, *memsize, arena) ;
    GB_Global_memtable_dump ( ) ;
    #endif

    if (p != NULL)
    { 
        // clear the block of memory with a parallel memset on the CPU
        int nthreads_max = GB_Context_nthreads_max ( ) ;
        GB_memset (p, 0, (*memsize), nthreads_max) ;
    }

    return (p) ;
}

//------------------------------------------------------------------------------
// GB_calloc_memory
//------------------------------------------------------------------------------

#if 0
void *GB_calloc_memory      // pointer to allocated block of memory
(
    uint64_t nitems,        // number of items to allocate
    uint64_t size_of_item,  // sizeof each item
    // input/output
    uint64_t *mem           // # of bytes actually allocated, and arena
)
#endif

GB_CALLBACK_CALLOC_MEMORY_PROTO (GB_calloc_memory)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (mem != NULL) ;

    void *p ;
    uint64_t memsize = 0 ;
    int arena = GB_arena (*mem) ;

    // make sure at least one item is allocated
    nitems = GB_IMAX (1, nitems) ;

    // make sure at least one byte is allocated
    size_of_item = GB_IMAX (1, size_of_item) ;

    bool ok = GB_uint64_multiply (&memsize, nitems, size_of_item) ;
    if (!ok || nitems > GB_NMAX || size_of_item > GB_NMAX)
    { 
        // overflow
        (*mem) = GB_mem (arena, 0) ;
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
            p = GB_calloc_helper (&memsize, arena) ;
        }

    }
    else
    { 

        //----------------------------------------------------------------------
        // normal use, in production
        //----------------------------------------------------------------------

        p = GB_calloc_helper (&memsize, arena) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    memsize = (p == NULL) ? 0 : memsize ;
    if (p != NULL)
    {
        MEMTABLE_ASSERT (memsize == GB_Global_memtable_memsize (p)) ;
        #ifdef GB_MEMTABLE_DEBUG
        if (arena != GB_Global_memtable_arena (p))
        {
            printf ("\narena: (%d,%d)!!\n", arena,
                GB_Global_memtable_arena (p)) ;
        }
        #endif
        MEMTABLE_ASSERT (arena == GB_Global_memtable_arena (p)) ;
    }
    (*mem) = GB_mem (arena, memsize) ;
    return (p) ;
}

