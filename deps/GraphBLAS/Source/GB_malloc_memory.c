//------------------------------------------------------------------------------
// GB_malloc_memory: wrapper for malloc
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A wrapper for malloc.  Space is not initialized.

#include "GB.h"

//------------------------------------------------------------------------------
// GB_malloc_helper:  use malloc to allocate an uninitialized memory block
//------------------------------------------------------------------------------

static inline void *GB_malloc_helper
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
    printf ("hard malloc %p %ld\n", p, *size) ; // MEMDUMP
    GB_Global_memtable_dump ( ) ;
    #endif

    return (p) ;
}

//------------------------------------------------------------------------------
// GB_malloc_memory
//------------------------------------------------------------------------------

GB_CALLBACK_MALLOC_MEMORY_PROTO (GB_malloc_memory)
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
            p = GB_malloc_helper (&size) ;
        }

    }
    else
    { 

        //----------------------------------------------------------------------
        // normal use, in production
        //----------------------------------------------------------------------

        p = GB_malloc_helper (&size) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*size_allocated) = (p == NULL) ? 0 : size ;
    ASSERT (GB_IMPLIES (p != NULL, size == GB_Global_memtable_size (p))) ;
    return (p) ;
}

