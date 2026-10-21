//------------------------------------------------------------------------------
// GB_set_arena: set the arena of a block of memory
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// CPU method for ensuring a pointer p is in a specified arena.
// CUDA will have its own method.

// If the new arena is not initialized or out of range, this method
// returns GrB_INVALID_VALUE.

#include "GB.h"

#define GB_FREE_ALL ;

GrB_Info GB_set_arena           // set arena of a block of memory
(
    // input/output:
    void **p_handle,            // block of memory to change
    uint64_t *p_mem_handle,     // memsize and arena of block of memory
    // input
    const int new_arena,        // arena to move to
    const uint64_t new_memsize, // new size of the block of memory
    const uint64_t n,           // # of bytes that must be copied
    const int nthreads          // max # of threads to use
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    if (p_handle == NULL || p_mem_handle == NULL || (*p_handle) == NULL)
    { 
        // nothing to do
        return (GrB_SUCCESS) ;
    }
    GB_OK (GB_check_arena (new_arena)) ;

    //--------------------------------------------------------------------------
    // get the current block
    //--------------------------------------------------------------------------

    void *p_old = (*p_handle) ;
    uint64_t p_old_mem = (*p_mem_handle) ;
    int old_arena = GB_arena (p_old_mem) ;
    #ifdef GB_DEBUG
    uint64_t old_memsize = GB_memsize (p_old_mem) ;
    ASSERT (new_memsize >= n) ;
    ASSERT (old_memsize >= n) ;
    #endif

    //--------------------------------------------------------------------------
    // quick return
    //--------------------------------------------------------------------------

    if (old_arena == new_arena)
    { 
        // nothing to do
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // allocate the new block in the new arena
    //--------------------------------------------------------------------------

    uint64_t p_new_mem = GB_mem (new_arena, 0) ;
    void *p_new = GB_MALLOC_MEMORY (new_memsize, sizeof (GB_void), &p_new_mem) ;
    if (p_new == NULL)
    { 
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // copy the data to the new block
    //--------------------------------------------------------------------------

    GB_memcpy (p_new, p_old, n, nthreads) ;

    //--------------------------------------------------------------------------
    // free the old block
    //--------------------------------------------------------------------------

    GB_FREE_MEMORY (&p_old, p_old_mem) ;

    //--------------------------------------------------------------------------
    // return results
    //--------------------------------------------------------------------------

    (*p_handle) = p_new ;
    (*p_mem_handle) = p_new_mem ;
    return (GrB_SUCCESS) ;
}

