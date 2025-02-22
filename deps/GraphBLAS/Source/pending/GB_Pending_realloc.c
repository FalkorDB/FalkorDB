//------------------------------------------------------------------------------
// GB_Pending_realloc: reallocate a list of pending tuples
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reallocate a list of pending tuples.  If it fails, the list is freed.

#include "pending/GB_Pending.h"

bool GB_Pending_realloc     // reallocate a list of pending tuples
(
    GrB_Matrix C,           // matrix to reallocate C->Pending for
    int64_t nnew,           // # of new tuples to accomodate
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (C != NULL) ;
    GB_Pending Pending = C->Pending ;
    ASSERT (Pending != NULL) ;

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

        size_t jsize = (C->j_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
        size_t isize = (C->i_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;

        bool ok1 = true ;
        bool ok2 = true ;
        bool ok3 = true ;

        GB_REALLOC_MEMORY (Pending->i, newsize, isize, &(Pending->i_size),
            &ok1) ;
        if (Pending->j != NULL)
        { 
            GB_REALLOC_MEMORY (Pending->j, newsize, jsize, &(Pending->j_size),
                &ok2) ;
        }
        size_t s = Pending->size ;
        if (Pending->x != NULL)
        { 
            GB_REALLOC_MEMORY (Pending->x, newsize, s, &(Pending->x_size),
                &ok3) ;
        }
        if (!ok1 || !ok2 || !ok3)
        { 
            // out of memory
            GB_Pending_free (&(C->Pending)) ;
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

