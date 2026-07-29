//------------------------------------------------------------------------------
// GB_arenas_will_wait: determine if all data is in A->data_arena
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

#ifdef comments_only

    // The CUDA branch tests will perform the following tests on its
    // input/output matrices:
    if (A != NULL && A->data_arena != GxB_ARENA_RMM)
    {
        cannot solve this problem with CUDA
    }
    if (!GB_shallow_arenas_ok (A))
    {
        cannot solve this problem with CUDA; it contains shallow components
        not in A->data_arena = GxB_ARENA_RMM
    }

    // the CUDA kernel will now work; but it must ensure it can access the data:
    if (GB_arenas_will_wait (A))
    {
        // wait on all data arenas to ensure they are in GxB_ARENA_RMM
        GB_wait_arenas (A)
    }
    GrB_Matrix A_alias = NULL ;
    if (GB_arena (A->header_mem) != GxB_ARENA_RMM)
    {
        // make an alias of A so the GPU can access the header of A
        GB_get_arena_alias (&A_alias, GxB_ARENA_RMM, A) ;
    }

    // ... do the CUDA kernel ... using A_alias if not-NULL, or A otherwise

    // if the A_alias was created (must always be done, even if the CUDA kernel
    // fails):
    if (A_alias != NULL)
    {
        GB_put_arena_alias (&A_alias, A) ;
    }

#endif

bool GB_arenas_will_wait
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
        // nothing to do
        return (false) ;
    }

    int A_data_arena = A->data_arena ;

    //--------------------------------------------------------------------------
    // check the arenas of A->[phbix]
    //--------------------------------------------------------------------------

    if (!A->p_shallow && A->p != NULL)
    { 
        if (GB_arena (A->p_mem) != A_data_arena) return (true) ;
    }

    if (!A->h_shallow && A->h != NULL)
    { 
        if (GB_arena (A->h_mem) != A_data_arena) return (true) ;
    }

    if (!A->b_shallow && A->b != NULL)
    { 
        if (GB_arena (A->b_mem) != A_data_arena) return (true) ;
    }

    if (!A->i_shallow && A->i != NULL)
    { 
        if (GB_arena (A->i_mem) != A_data_arena) return (true) ;
    }

    if (!A->x_shallow && A->x != NULL)
    { 
        if (GB_arena (A->x_mem) != A_data_arena) return (true) ;
    }

    //--------------------------------------------------------------------------
    // check the arenas of the hyperhash
    //--------------------------------------------------------------------------

    if (!A->Y_shallow && A->Y != NULL)
    { 
        if (A->Y->data_arena != A_data_arena) return (true) ;
        if (GB_arenas_will_wait (A->Y)) return (true) ;
    }

    //--------------------------------------------------------------------------
    // check the arenas of the Pending tuples
    //--------------------------------------------------------------------------

    GB_Pending Pending = A->Pending ;
    if (Pending != NULL)
    { 
        if (GB_arena (Pending->header_mem) != A_data_arena) return (true) ;

        if (Pending->i != NULL)
        { 
            if (GB_arena (Pending->i_mem) != A_data_arena) return (true) ;
        }

        if (Pending->j != NULL)
        { 
            if (GB_arena (Pending->j_mem) != A_data_arena) return (true) ;
        }

        if (Pending->x != NULL)
        { 
            if (GB_arena (Pending->x_mem) != A_data_arena) return (true) ;
        }
    }

    //--------------------------------------------------------------------------
    // check header_arena memory
    //--------------------------------------------------------------------------

    int A_header_arena = GB_arena (A->header_mem) ;
    if (A->user_name != NULL)
    { 
        if (GB_arena (A->user_name_mem) != A_header_arena) return (true) ;
    }
    if (A->logger != NULL)
    { 
        if (GB_arena (A->logger_mem) != A_header_arena) return (true) ;
    }

    //--------------------------------------------------------------------------
    // all data arenas of components of A match A->data_arena
    //--------------------------------------------------------------------------

    return (false) ;
}

