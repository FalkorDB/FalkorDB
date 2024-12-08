//------------------------------------------------------------------------------
// GB_realloc_memory: wrapper for realloc
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A wrapper for realloc.

// If p is non-NULL on input, it points to a previously allocated object of
// size at least nitems_old * size_of_item.  The object is reallocated to be of
// size at least nitems_new * size_of_item.  If p is NULL on input, then a new
// object of that size is allocated.  On success, a pointer to the new object
// is returned, and ok is returned as true.  If the allocation fails, ok is set
// to false and a pointer to the old (unmodified) object is returned.

// The actual size_allocated on input can differ from nitems_old*size_of_item,
// and the size_allocated on output can be larger than nitems_new*size_of_item,
// if the size_allocated is rounded up to the nearest power of two.

// Usage:

//      p = GB_realloc_memory (nitems_new, size_of_item, p,
//              &size_allocated, &ok)
//      if (ok)
//      {
//          p points to a block of at least nitems_new*size_of_item bytes and
//          the first part, of size min(nitems_new,nitems_old)*size_of_item,
//          has the same content as the old memory block if it was present.
//      }
//      else
//      {
//          p points to the old block, and size_allocated is left
//          unchanged.  This case never occurs if nitems_new < nitems_old.
//      }
//      on output, size_allocated is set to the actual size of the block of
//      memory

#include "GB.h"

void *GB_realloc_memory     // pointer to reallocated block of memory, or
                            // to original block if the reallocation failed.
(
    size_t nitems_new,      // new number of items in the object
    size_t size_of_item,    // size of each item
    // input/output
    void *p,                // old object to reallocate
    size_t *size_allocated, // # of bytes actually allocated
    // output
    bool *ok                // true if successful, false otherwise
)
{

    //--------------------------------------------------------------------------
    // malloc a new block if p is NULL on input
    //--------------------------------------------------------------------------

    if (p == NULL)
    { 
        p = GB_malloc_memory (nitems_new, size_of_item, size_allocated) ;
        (*ok) = (p != NULL) ;
        return (p) ;
    }

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // make sure at least one byte is allocated
    size_of_item = GB_IMAX (1, size_of_item) ;

    size_t oldsize_allocated = (*size_allocated) ;
    ASSERT (oldsize_allocated == GB_Global_memtable_size (p)) ;

    // make sure at least one item is allocated
    size_t nitems_old = oldsize_allocated / size_of_item ;
    nitems_new = GB_IMAX (1, nitems_new) ;

    size_t newsize, oldsize ;
    (*ok) = GB_size_t_multiply (&newsize, nitems_new, size_of_item)
         && GB_size_t_multiply (&oldsize, nitems_old, size_of_item) ;

    if (!(*ok) || (((uint64_t) nitems_new) > GB_NMAX)
               || (((uint64_t) size_of_item) > GB_NMAX))
    { 
        // overflow
        (*ok) = false ;
        return (p) ;
    }

    //--------------------------------------------------------------------------
    // check for quick return
    //--------------------------------------------------------------------------

    if ((newsize == oldsize)
        || (newsize < oldsize && newsize >= oldsize_allocated/2)
        || (newsize > oldsize && newsize <= oldsize_allocated))
    { 
        // If the block does not change, or is shrinking but only by a small
        // amount, or is growing but still fits inside the existing block,
        // then leave the block as-is.
        (*ok) = true ;
        return (p) ;
    }

    //--------------------------------------------------------------------------
    // reallocate the memory, or use malloc/memcpy/free
    //--------------------------------------------------------------------------

    void *pnew = NULL ;
    size_t newsize_allocated = GB_IMAX (newsize, 8) ;
    if (!GB_Global_have_realloc_function ( ))
    {

        //----------------------------------------------------------------------
        // no realloc function: use malloc/memcpy/free
        //----------------------------------------------------------------------

        // allocate the new space
        pnew = GB_malloc_memory (nitems_new, size_of_item, &newsize_allocated) ;
        // copy over the data from the old block to the new block
        if (pnew != NULL)
        { 
            // copy from the old to new with a parallel memcpy
            int nthreads_max = GB_Context_nthreads_max ( ) ;
            GB_memcpy (pnew, p, GB_IMIN (oldsize, newsize), nthreads_max) ;
            // free the old block
            GB_free_memory (&p, oldsize_allocated) ;
        }
    }
    else
    {

        //----------------------------------------------------------------------
        // use realloc
        //----------------------------------------------------------------------

        bool pretend_to_fail = false ;
        if (GB_Global_malloc_tracking_get ( ) && GB_Global_malloc_debug_get ( ))
        {
            pretend_to_fail = GB_Global_malloc_debug_count_decrement ( ) ;
        }
        if (!pretend_to_fail)
        { 
            #ifdef GB_MEMDUMP
            printf ("hard realloc %p oldsize %ld newsize %ld\n",    // MEMDUMP
                p, oldsize_allocated, newsize_allocated) ;
            #endif
            pnew = GB_Global_realloc_function (p, newsize_allocated) ;
            #ifdef GB_MEMDUMP
            GB_Global_memtable_dump ( ) ;
            #endif
        }
    }

    //--------------------------------------------------------------------------
    // check if successful and return result
    //--------------------------------------------------------------------------

    if (pnew == NULL)
    {
        // realloc failed
        if (newsize < oldsize)
        { 
            // the attempt to reduce the size of the block failed, but the old
            // block is unchanged.  So pretend to succeed, but do not change
            // size_allocated since it must reflect the actual size of the
            // block.
            (*ok) = true ;
        }
        else
        { 
            // out of memory.  the old block is unchanged
            (*ok) = false ;
        }
    }
    else
    { 
        // realloc succeeded
        p = pnew ;
        (*ok) = true ;
        (*size_allocated) = newsize_allocated ;
    }

    return (p) ;
}

