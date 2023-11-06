//------------------------------------------------------------------------------
// GB_calloc_memory: wrapper for calloc (actually uses malloc and memset)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A wrapper for calloc.  Space is set to zero.

#include "GB.h"

//------------------------------------------------------------------------------
// GB_calloc_helper:  malloc/memset to allocate an initialized block
//------------------------------------------------------------------------------

static inline void *GB_calloc_helper
(
    // input/output:
    size_t *size            // on input: # of bytes requested
                            // on output: # of bytes actually allocated
)
{
    void *p = NULL ;

    // make sure the block is at least 8 bytes in size
    (*size) = GB_IMAX (*size, 8) ;

    p = GB_Global_malloc_function (*size) ;

    #ifdef GB_MEMDUMP
    printf ("hard calloc %p %ld\n", p, *size) ; // MEMDUMP
    GB_Global_memtable_dump ( ) ;
    #endif

    if (p != NULL)
    { 
        // clear the block of memory with a parallel memset
        int nthreads_max = GB_Context_nthreads_max ( ) ;
        GB_memset (p, 0, (*size), nthreads_max) ;
    }

    return (p) ;
}

//------------------------------------------------------------------------------
// GB_calloc_memory
//------------------------------------------------------------------------------

void *GB_calloc_memory      // pointer to allocated block of memory
(
    size_t nitems,          // number of items to allocate
    size_t size_of_item,    // sizeof each item
    // output
    size_t *size_allocated  // # of bytes actually allocated
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (size_allocated != NULL) ;

    void *p ;
    size_t size ;

    // make sure at least one item is allocated
    nitems = GB_IMAX (1, nitems) ;

    // make sure at least one byte is allocated
    size_of_item = GB_IMAX (1, size_of_item) ;

    bool ok = GB_size_t_multiply (&size, nitems, size_of_item) ;
    if (!ok || (((uint64_t) nitems) > GB_NMAX)
            || (((uint64_t) size_of_item) > GB_NMAX))
    { 
        // overflow
        (*size_allocated) = 0 ;
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
            p = GB_calloc_helper (&size) ;
        }

    }
    else
    { 

        //----------------------------------------------------------------------
        // normal use, in production
        //----------------------------------------------------------------------

        p = GB_calloc_helper (&size) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*size_allocated) = (p == NULL) ? 0 : size ;
    ASSERT (GB_IMPLIES (p != NULL, size == GB_Global_memtable_size (p))) ;
    return (p) ;
}

