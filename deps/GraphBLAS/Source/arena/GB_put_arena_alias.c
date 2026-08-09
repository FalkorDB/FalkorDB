//------------------------------------------------------------------------------
// GB_put_arena_alias: create an alias of a matrix in a new header arena
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// fixme for CUDA:  this method is a draft.  It will be used for CUDA.  It is
// not yet used by any methods, even when CUDA is enabled.

// GB_put_arena_alias restores the (possibly revised) content of the header
// of an alias C of the matrix A, and frees the header for C.

// This method always succeeds.

#include "GB.h"

void GB_put_arena_alias
(
    // input/outputs
    GrB_Matrix *Chandle,    // alias of A to be freed; NULL on output
    const GrB_Matrix A      // updated with any revisions in the alias header C
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Chandle != NULL) ;
    ASSERT ((*Chandle) != NULL) ;
    ASSERT_MATRIX_OK (*Chandle, "&C for put_arena_alias", GB0) ;

    //--------------------------------------------------------------------------
    // copy the alias header C back in A, and free C
    //--------------------------------------------------------------------------

    // get the header_mem's of A and C
    uint64_t A_header_mem = A->header_mem ;
    uint64_t C_header_mem = (*Chandle)->header_mem ;

    // copy the entire alias C back into the primary matrix A
    memcpy (A, *Chandle, sizeof (struct GB_Matrix_opaque)) ;

    // restore the true header of A (was overwritten by the memcpy above)
    A->header_mem = A_header_mem ;

    // free the alias header C
    GB_FREE_MEMORY (Chandle, C_header_mem) ;
    ASSERT ((*Chandle) == NULL) ;
}

