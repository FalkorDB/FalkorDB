//------------------------------------------------------------------------------
// GB_convert_full_to_sparse: convert a matrix from full to sparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed.  Only one variant possible.

#include "GB.h"

GrB_Info GB_convert_full_to_sparse      // convert matrix from full to sparse
(
    GrB_Matrix A                // matrix to convert from full to sparse
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converting full to sparse", GB0) ;
    ASSERT (GB_IS_FULL (A) || GB_nnz_max (A) == 0) ;
    ASSERT (!GB_IS_BITMAP (A)) ;
    ASSERT (!GB_IS_SPARSE (A)) ;
    ASSERT (!GB_IS_HYPERSPARSE (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    //--------------------------------------------------------------------------
    // allocate A->p and A->i
    //--------------------------------------------------------------------------

    int64_t avdim = A->vdim ;
    int64_t avlen = A->vlen ;
    int64_t anz = GB_nnz_full (A) ;
    GB_BURBLE_N (anz, "(full to sparse) ") ;
    int64_t *restrict Ap = NULL ; size_t Ap_size = 0 ;
    int64_t *restrict Ai = NULL ; size_t Ai_size = 0 ;
    Ap = GB_MALLOC (avdim+1, int64_t, &Ap_size) ;
    Ai = GB_MALLOC (anz, int64_t, &Ai_size) ;
    if (Ap == NULL || Ai == NULL)
    { 
        // out of memory
        GB_FREE (&Ap, Ap_size) ;
        GB_FREE (&Ai, Ai_size) ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    A->p = Ap ; A->p_size = Ap_size ;
    A->i = Ai ; A->i_size = Ai_size ;
    A->plen = avdim ;
    A->nvec = avdim ;
    A->nvec_nonempty = (avlen == 0) ? 0 : avdim ;
    A->nvals = anz ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (anz, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // fill the A->p and A->i pattern
    //--------------------------------------------------------------------------

    int64_t k ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (k = 0 ; k <= avdim ; k++)
    { 
        Ap [k] = k * avlen ;
    }

    int64_t p ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (p = 0 ; p < anz ; p++)
    { 
        Ai [p] = p % avlen ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converted from full to sparse", GB0) ;
    ASSERT (GB_IS_SPARSE (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    return (GrB_SUCCESS) ;
}

