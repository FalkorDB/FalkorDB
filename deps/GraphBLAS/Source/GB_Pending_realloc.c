//------------------------------------------------------------------------------
// GB_Pending_realloc: reallocate a list of pending tuples
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reallocate a list of pending tuples.  If it fails, the list is freed.

#include "GB_Pending.h"

bool GB_Pending_realloc     // reallocate a list of pending tuples
(
    GB_Pending *PHandle,    // Pending tuple list to reallocate
    int64_t nnew,           // # of new tuples to accomodate
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (PHandle != NULL) ;
    GB_Pending Pending = (*PHandle) ;

    //--------------------------------------------------------------------------
    // ensure the list can hold at least nnew more tuples
    //--------------------------------------------------------------------------

    int64_t newsize = nnew + Pending->n ;

    if (newsize > Pending->nmax)
    {

        //----------------------------------------------------------------------
        // double the size if the list is not large enough
        //----------------------------------------------------------------------

        newsize = GB_IMAX (newsize, 2 * Pending->nmax) ;

        //----------------------------------------------------------------------
        // reallocate the i,j,x arrays
        //----------------------------------------------------------------------

        bool ok1 = true ;
        bool ok2 = true ;
        bool ok3 = true ;

        GB_REALLOC (Pending->i, newsize, int64_t, &(Pending->i_size), &ok1) ;
        if (Pending->j != NULL)
        { 
            GB_REALLOC (Pending->j, newsize, int64_t, &(Pending->j_size), &ok2);
        }
        size_t s = Pending->size ;
        if (Pending->x != NULL)
        { 
            GB_REALLOC (Pending->x, newsize*s, GB_void, &(Pending->x_size),
                &ok3) ;
        }
        if (!ok1 || !ok2 || !ok3)
        { 
            // out of memory
            GB_Pending_free (PHandle) ;
            return (false) ;
        }

        //----------------------------------------------------------------------
        // record the new size of the Pending tuple list
        //----------------------------------------------------------------------

        Pending->nmax = newsize ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (true) ;
}

