//------------------------------------------------------------------------------
// GB_conform_hyper: conform a sparse matrix to its desired hypersparse format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input matrix must be sparse or hypersparse, and it may be left as-is,
// or converted to sparse/hypersparse.

// The input matrix can have shallow A->p and/or A->h components.  If the
// hypersparsity is changed, these components are no longer shallow.

#include "GB.h"

#define GB_FREE_ALL ;

GrB_Info GB_conform_hyper       // conform a matrix to sparse/hypersparse
(
    GrB_Matrix A,               // matrix to conform
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_SUCCESS ;
    ASSERT_MATRIX_OK (A, "A to conform_hyper", GB0) ;
    ASSERT (!GB_IS_FULL (A)) ;
    ASSERT (!GB_IS_BITMAP (A)) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_PENDING_OK (A)) ;

    //--------------------------------------------------------------------------
    // convert to sparse or hypersparse
    //--------------------------------------------------------------------------

    if (A->nvec_nonempty < 0)
    { 
        // A->nvec_nonempty is used to select sparse vs hypersparse
        A->nvec_nonempty = GB_nvec_nonempty (A) ;
    }

    if (A->h == NULL && GB_convert_sparse_to_hyper_test (A->hyper_switch,
        A->nvec_nonempty, A->vdim)) // A->nvec_nonempty used here
    { 
        // A is sparse but should be converted to hypersparse
        GB_OK (GB_convert_sparse_to_hyper (A, Werk)) ;
    }
    else if (A->h != NULL && GB_convert_hyper_to_sparse_test (A->hyper_switch,
        A->nvec_nonempty, A->vdim)) // A->nvec_nonempty used here
    { 
        // A is hypersparse but should be converted to sparse
        GB_OK (GB_convert_hyper_to_sparse (A, true)) ;
    }
    else
    { 
        // leave the matrix as-is
        ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A conform_hyper result", GB0) ;
    return (GrB_SUCCESS) ;
}

