//------------------------------------------------------------------------------
// GB_get_arena_alias: create an alias of a matrix in a new header arena
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// fixme for CUDA:  this method is a draft.  It will be used for CUDA.  It is
// not yet used by any methods, even when CUDA is enabled.

// GB_get_arena_alias creates a new matrix C that is an alias for the input
// matrix A, where all the pointers to data in the C matrix point to the
// identical content in A.  Only C->header_mem and A->header_mem can differ.

// While the alias exists, A should not be used via its own header, since any
// GraphBLAS method might make changes to C (if it has pending work, or if A
// is the output matrix of any GraphBLAS operation).  Only the alias C should
// be used.

// This alias cannot be freed with GrB_Matrix_free, since that would also free
// the data content of A, leaving A invalid.  Instead, once the alias is no
// longer needed, the header content of the alias C should be copied back into
// A, and the header for C freed, using GB_put_arena_alias (&C, A).  That is:

/*
    GrB_Matrix C = NULL ;
    GB_get_arena_alias (&C, new_header_arena, A) ;
    ...
    use C here, not A, making any changes to C as if A is being modified
    ...
    GB_put_arena_alias (&C, A) ;
    // C is now NULL, and A contains any modifications made via C
*/

// Aliased matrices are never returned to the user application.

#include "GB.h"

#define GB_FREE_ALL ;

GrB_Info GB_get_arena_alias
(
    // output
    GrB_Matrix *Chandle,    // output matrix, (*Chandle) is NULL on input
    // inputs
    const int new_header_arena, // arena for C header
    const GrB_Matrix A      // input matrix
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (Chandle != NULL) ;
    ASSERT ((*Chandle) == NULL) ;
    ASSERT_MATRIX_OK (A, "A for get_arena_alias", GB0) ;
    GB_OK (GB_check_arena (new_header_arena)) ;

    //--------------------------------------------------------------------------
    // copy the header
    //--------------------------------------------------------------------------

    // create a new alias header C
    GB_OK (GB_matrix_header_new (Chandle, new_header_arena, A->data_arena)) ;

    // save the C->header_mem
    uint64_t C_header_mem = (*Chandle)->header_mem ;

    // copy the entire header of the primary matrix A into the alias C
    memcpy (*Chandle, A, sizeof (struct GB_Matrix_opaque)) ;

    // restore the true C->header_mem (was overwritten by the memcpy above)
    (*Chandle)->header_mem = C_header_mem ;
    return (GrB_SUCCESS) ;
}

