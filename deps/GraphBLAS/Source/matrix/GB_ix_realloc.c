//------------------------------------------------------------------------------
// GB_ix_realloc: reallocate a sparse/hyper matrix to hold a given # of entries
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reallocates A->x and A->i to the requested size, preserving the existing
// content of A->x and A->i.  Preserves pending tuples and zombies, if any.
// A->j_is_32 and A->i_is_32 are unchanged since the matrix dimensions do not
// change, and thus A->Y is not modified.

// If nzmax_new is too large for the current value of A->p_is_32, then A->p
// is converted to 64-bit integers and A->p_is_32 is set true.  The content of
// A->p is preserved.

// Thus, this method typically changes the A->x and A->i pointers, and may
// change A->p if needed.

// If this method runs out of memory, the matrix is unchanged.

#define GB_FREE_ALL ;

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

    // Full and bitmap matrices never have pending work, so
    // this function is only called for hypersparse and sparse matrices.
    GrB_Info info ;
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
    // change A->p_is_32 and reallocate A->p if required
    //--------------------------------------------------------------------------

    if (!GB_valid_p_is_32 (A->p_is_32, nzmax_new))
    { 
        // convert A->p to 64-bit; do not change A->j_is_32 or A->i_is_32; note
        // that GB_convert_int does not validate the new integer settings,
        // since the # of entries in A is changing.
        GB_OK (GB_convert_int (A, false, A->j_is_32, A->i_is_32, false)) ;
    }

    //--------------------------------------------------------------------------
    // reallocate A->i (does not change A->i_is_32)
    //--------------------------------------------------------------------------

    size_t nzmax_new1 = GB_IMAX (nzmax_new, 1) ;
    bool ok1 = true, ok2 = true ;
    size_t isize = A->i_is_32 ? sizeof (int32_t) : sizeof (int64_t) ;
    GB_REALLOC_MEMORY (A->i, nzmax_new1, isize, &(A->i_size), &ok1) ;

    //--------------------------------------------------------------------------
    // reallocate A->x
    //--------------------------------------------------------------------------

    size_t asize = A->type->size ;
    if (A->iso)
    { 
        // shrink A->x so it holds a single entry
        GB_REALLOC_MEMORY (A->x, 1, asize, &(A->x_size), &ok2) ;
    }
    else
    { 
        // reallocate A->x from its current size to nzmax_new1 entries
        GB_REALLOC_MEMORY (A->x, nzmax_new1, asize, &(A->x_size), &ok2) ;
    }
    bool ok = ok1 && ok2 ;

    //--------------------------------------------------------------------------
    // check and return result
    //--------------------------------------------------------------------------

    // The matrix is always left in a valid state.  If the reallocation fails
    // it just won't have the requested size.
    if (!ok)
    { 
        return (GrB_OUT_OF_MEMORY) ;
    }

    return (GrB_SUCCESS) ;
}

