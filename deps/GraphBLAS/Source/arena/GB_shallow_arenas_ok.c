//------------------------------------------------------------------------------
// GB_shallow_arenas_ok: check if A has shallow components not in A->data_arena
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// fixme for CUDA:  this method is a draft.  It will be used for CUDA.  It is
// not yet used by any methods, even when CUDA is enabled.

// Shallow components of A are not moved by GB_set_arenas.  CUDA cannot access
// these if they are not in a data_arena using Rapids.  This method returns
// false if A has any shallow components not in A->data_arena.  If A is NULL,
// or if A has no shallow components, or if all shallow components are in
// A->data_arena, this method returns true.

#include "GB.h"

bool GB_shallow_arenas_ok
(
    // input/output:
    GrB_Matrix A
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (A == NULL)
    {
        // nothing to do; all is fine
        return (true) ;
    }

    int A_data_arena = A->data_arena ;

    //--------------------------------------------------------------------------
    // check the arenas of A->[phbix]
    //--------------------------------------------------------------------------

    if (A->p_shallow && A->p != NULL)
    {
        if (GB_arena (A->p_mem) != A_data_arena) return (false) ;
    }

    if (A->h_shallow && A->h != NULL)
    {
        if (GB_arena (A->h_mem) != A_data_arena) return (false) ;
    }

    if (A->b_shallow && A->b != NULL)
    {
        if (GB_arena (A->b_mem) != A_data_arena) return (false) ;
    }

    if (A->i_shallow && A->i != NULL)
    {
        if (GB_arena (A->i_mem) != A_data_arena) return (false) ;
    }

    if (A->x_shallow && A->x != NULL)
    {
        if (GB_arena (A->x_mem) != A_data_arena) return (false) ;
    }

    //--------------------------------------------------------------------------
    // check the arenas of the hyperhash
    //--------------------------------------------------------------------------

    if (A->Y != NULL)
    {
        if (A->Y_shallow && A->Y->data_arena != A_data_arena) return (false) ;
        if (!GB_shallow_arenas_ok (A->Y)) return (false) ;
    }

    //--------------------------------------------------------------------------
    // any shallow components of A match A->data_arena
    //--------------------------------------------------------------------------

    return (true) ;
}

