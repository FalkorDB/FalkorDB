//------------------------------------------------------------------------------
// GB_ix_realloc: reallocate a sparse/hyper matrix to hold a given # of entries
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Does not modify A->p.  Reallocates A->x and A->i to the requested size,
// preserving the existing content of A->x and A->i.  Preserves pending tuples
// and zombies, if any.

#include "GB.h"

GrB_Info GB_ix_realloc      // reallocate space in a matrix
(
    GrB_Matrix A,               // matrix to allocate space for
    const int64_t nzmax_new     // new number of entries the matrix can hold
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // This method is used only by GB_ix_resize, which itself is used only by
    // GB_wait.  Full and bitmap matrices never have pending work, so
    // this function is only called for hypersparse and sparse matrices.
    ASSERT (!GB_IS_FULL (A)) ;
    ASSERT (!GB_IS_BITMAP (A)) ;
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;

    // A->p has been allocated but might not be initialized.  GB_matvec_check
    // fails in this case.  Thus, ASSERT_MATRIX_OK (A, "A", ...) ;  cannot be
    // used here.
    ASSERT (A != NULL && A->p != NULL) ;
    ASSERT (!A->i_shallow && !A->x_shallow) ;

    // This function tolerates pending tuples, zombies, and jumbled matrices.
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_PENDING_OK (A)) ;

    if (nzmax_new > GB_NMAX)
    { 
        // problem too large
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // reallocate the space
    //--------------------------------------------------------------------------

    size_t nzmax_new1 = GB_IMAX (nzmax_new, 1) ;
    bool ok1 = true, ok2 = true ;
    GB_REALLOC (A->i, nzmax_new1, int64_t, &(A->i_size), &ok1) ;
    size_t asize = A->type->size ;
    if (A->iso)
    { 
        // shrink A->x so it holds a single entry
        GB_REALLOC (A->x, asize, GB_void, &(A->x_size), &ok2) ;
    }
    else
    { 
        // reallocate A->x from its current size to nzmax_new1 entries
        GB_REALLOC (A->x, nzmax_new1*asize, GB_void, &(A->x_size), &ok2) ;
    }
    bool ok = ok1 && ok2 ;

    // The matrix is always left in a valid state.  If the reallocation fails
    // it just won't have the requested size (and ok is false in this case).
    if (!ok)
    { 
        return (GrB_OUT_OF_MEMORY) ;
    }

    return (GrB_SUCCESS) ;
}

