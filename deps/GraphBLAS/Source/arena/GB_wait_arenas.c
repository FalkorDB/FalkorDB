//------------------------------------------------------------------------------
// GB_wait_arenas:  align the data with A->data_arena
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The header of A is unchanged.  All data content is moved into the
// A->data_arena.  A->user_name and A->logger are moved into the same arena as
// the header of A.  Any phybix shallow components of A remain unchanged.  A
// may have pending work (zombies, pending tuples, jumbled, or hyperhash
// needed); this is left unmodified.  All pending work remains undone.

#include "GB.h"

#define GB_FREE_ALL GB_phybix_free (A) ;

GrB_Info GB_wait_arenas         // align data with A->data_arena
(
    GrB_Matrix A                // input/output matrix
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    if (A == NULL)
    { 
        // nothing to do
        return (GrB_SUCCESS) ;
    }

    int header_arena = GB_arena (A->header_mem) ;
    int data_arena = A->data_arena ;

    ASSERT_MATRIX_OK (A, "A to wait_arenas: start", GB0_Z) ;

    //--------------------------------------------------------------------------
    // align the data arenas with A->data_arena
    //--------------------------------------------------------------------------

    GrB_Matrix T = A ;
    GB_OK (GB_set_arenas (&T, header_arena, data_arena)) ;
    ASSERT (T == A) ;       // GB_set_arenas will not change the header of A

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A wait_arenas: done", GB0_Z) ;
    ASSERT (!GB_arenas_will_wait (A)) ;
    return (GrB_SUCCESS) ;
}

