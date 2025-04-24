//------------------------------------------------------------------------------
// LAGraph_Realloc: wrapper for realloc
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

// If p is non-NULL on input, it points to a previously allocated object of
// size at least nitems_old * size_of_item.  The object is reallocated to be of
// size at least nitems_new * size_of_item.  If p is NULL on input, then a new
// object of that size is allocated.  On success, a pointer to the new object
// is returned, and ok is returned as true.  If the allocation fails, ok is set
// to false and a pointer to the old (unmodified) object is returned.

// Usage:

//  int status = LAGraph_Realloc (&p, nitems_new, nitems_old, size_of_item, msg)
//  if (status == GrB_SUCCESS)
//  {
//      p points to a block of at least nitems_new*size_of_item bytes and
//      the first part, of size min(nitems_new,nitems_old)*size_of_item,
//      has the same content as the old memory block if it was present.
//  }
//  else
//  {
//      p points to the old block, unchanged.  This case never occurs if
//      nitems_new < nitems_old.
//  }

#include "LG_internal.h"

int LAGraph_Realloc
(
    // input/output:
    void **p,               // old block to reallocate
    // input:
    size_t nitems_new,      // new number of items in the object
    size_t nitems_old,      // old number of items in the object
    size_t size_of_item,    // size of each item
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    LG_ASSERT (p != NULL, GrB_NULL_POINTER) ;

    //--------------------------------------------------------------------------
    // malloc a new block if p is NULL on input
    //--------------------------------------------------------------------------

    if ((*p) == NULL)
    {
        LG_TRY (LAGraph_Malloc (p, nitems_new, size_of_item, msg)) ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // make sure at least one item is allocated
    nitems_old = LAGRAPH_MAX (1, nitems_old) ;
    nitems_new = LAGRAPH_MAX (1, nitems_new) ;

    // make sure at least one byte is allocated
    size_of_item = LAGRAPH_MAX (1, size_of_item) ;

    size_t newsize, oldsize ;
    bool ok = LG_Multiply_size_t (&newsize, nitems_new, size_of_item)
           && LG_Multiply_size_t (&oldsize, nitems_old, size_of_item) ;
    if (!ok)
    {
        // overflow
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // reallocate an existing block to accommodate the change in # of items
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // check for quick return
    //--------------------------------------------------------------------------

    if (newsize == oldsize)
    {
        // If the block does not change, or is shrinking but only by a small
        // amount, or is growing but still fits inside the existing block,
        // then leave the block as-is.
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // reallocate the memory, or use malloc/memcpy/free
    //--------------------------------------------------------------------------

    void *pnew = NULL ;

    if (LAGraph_Realloc_function == NULL)
    {

        //----------------------------------------------------------------------
        // use malloc/memcpy/free
        //----------------------------------------------------------------------

        // allocate the new space
        LG_TRY (LAGraph_Malloc (&pnew, nitems_new, size_of_item, msg)) ;
        // copy over the data from the old block to the new block
        // copy from the old to the new space
        memcpy (pnew, *p, LAGRAPH_MIN (oldsize, newsize)) ;
        // free the old space
        LG_TRY (LAGraph_Free (p, msg)) ;
        (*p) = pnew ;

    }
    else
    {

        //----------------------------------------------------------------------
        // use realloc
        //----------------------------------------------------------------------

        pnew = LAGraph_Realloc_function (*p, newsize) ;
        if (pnew == NULL) return (GrB_OUT_OF_MEMORY) ;
        (*p) = pnew ;
    }

    return (GrB_SUCCESS) ;
}
