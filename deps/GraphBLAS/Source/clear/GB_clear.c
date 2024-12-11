//------------------------------------------------------------------------------
// GB_clear: clears the content of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// All content of A is freed (or removed if shallow) and new A->p and A->h
// content is created.  This puts the matrix A in the same initialized state it
// had after GrB_Matrix_new (&A, ...), with A->magic == GB_MAGIC to denote a
// valid, initialized matrix, with nnz(A) equal to zero.  The dimensions, type,
// and CSR/CSC format are unchanged.  The hypersparsity of the newly empty
// matrix A is determined by the A->hyper_switch for the matrix.  The matrix is
// valid.

// However, if this method runs out of memory, and the A->p and A->h structure
// cannot be recreated, then all content of the matrix is freed or removed, and
// the matrix A is left in an invalid state (A->magic == GB_MAGIC2).  Only the
// header is left.

// A is first converted to sparse or hypersparse, and then conformed via
// GB_conform.  If A->sparsity_control disables the sparse and hypersparse
// structures, A is converted bitmap instead.

#include "GB.h"

GrB_Info GB_clear           // clear a matrix, type and dimensions unchanged
(
    GrB_Matrix A,           // matrix to clear
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (A != NULL) ;
    ASSERT (A->magic == GB_MAGIC || A->magic == GB_MAGIC2) ;

    // zombies and pending tuples have no effect; about to delete them anyway
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_PENDING_OK (A)) ;

    //--------------------------------------------------------------------------
    // clear the content of A if bitmap
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    int sparsity_control = GB_sparsity_control (A->sparsity_control, A->vdim) ;
    if (((sparsity_control & (GxB_SPARSE + GxB_HYPERSPARSE)) == 0)
        && GB_IS_BITMAP (A))
    { 
        // A should remain bitmap
        GB_memset (A->b, 0, GB_nnz_held (A), nthreads_max) ;
        A->nvals = 0 ;
        A->magic = GB_MAGIC ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // clear the content of A
    //--------------------------------------------------------------------------

    // free all content
    GB_phybix_free (A) ;

    // no more zombies or pending tuples
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    //--------------------------------------------------------------------------
    // allocate new A->p and A->h components
    //--------------------------------------------------------------------------

    // By default, an empty matrix with n > 1 vectors is held in hypersparse
    // form.  A GrB_Matrix with n <= 1, or a GrB_Vector (with n == 1) is always
    // non-hypersparse.  If A->hyper_switch is negative, A will be always be
    // non-hypersparse.

    if (GB_convert_hyper_to_sparse_test (A->hyper_switch, 0, A->vdim))
    {

        //----------------------------------------------------------------------
        // A is sparse
        //----------------------------------------------------------------------

        int64_t plen = A->vdim ;
        A->nvec = plen ;
        A->plen = plen ;
        A->p = GB_MALLOC (plen+1, int64_t, &(A->p_size)) ;
        ASSERT (A->h == NULL) ;
        if (A->p == NULL)
        { 
            // out of memory
            GB_phybix_free (A) ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        GB_memset (A->p, 0, (plen+1) * sizeof (int64_t), nthreads_max) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // A is hypersparse
        //----------------------------------------------------------------------

        int64_t plen = GB_IMIN (1, A->vdim) ;
        A->nvec = 0 ;
        A->plen = plen ;
        A->p = GB_MALLOC (plen+1, int64_t, &(A->p_size)) ;
        A->h = GB_MALLOC (plen  , int64_t, &(A->h_size)) ;
        if (A->p == NULL || A->h == NULL)
        { 
            // out of memory
            GB_phybix_free (A) ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        A->p [0] = 0 ;
        if (plen > 0)
        { 
            A->p [1] = 0 ;
            A->h [0] = 0 ;
        }
    }

    A->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // conform A to its desired sparsity 
    //--------------------------------------------------------------------------

    return (GB_conform (A, Werk)) ;
}

