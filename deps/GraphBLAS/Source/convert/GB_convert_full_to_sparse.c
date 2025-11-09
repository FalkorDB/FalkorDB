//------------------------------------------------------------------------------
// GB_convert_full_to_sparse: convert a matrix from full to sparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GB_convert_full_to_sparse      // convert matrix from full to sparse
(
    GrB_Matrix A,               // matrix to convert from full to sparse
    GB_Werk Werk
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

    bool Ap_is_32, Aj_is_32, Ai_is_32 ;
    GB_determine_pji_is_32 (&Ap_is_32, &Aj_is_32, &Ai_is_32,
        GxB_AUTO_SPARSITY, anz, avlen, avdim, Werk) ;

    void *Ap = NULL ; size_t Ap_size = 0 ;
    void *Ai = NULL ; size_t Ai_size = 0 ;

    size_t psize = (Ap_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t isize = (Ai_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;

    Ap = GB_MALLOC_MEMORY (avdim+1, psize, &Ap_size) ;
    Ai = GB_MALLOC_MEMORY (anz, isize, &Ai_size) ;
    if (Ap == NULL || Ai == NULL)
    { 
        // out of memory
        GB_FREE_MEMORY (&Ap, Ap_size) ;
        GB_FREE_MEMORY (&Ai, Ai_size) ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    A->p = Ap ; A->p_size = Ap_size ;
    A->i = Ai ; A->i_size = Ai_size ;
    A->p_is_32 = Ap_is_32 ;
    A->j_is_32 = Aj_is_32 ;
    A->i_is_32 = Ai_is_32 ;
    A->plen = avdim ;
    A->nvec = avdim ;
//  A->nvec_nonempty = (avlen == 0) ? 0 : avdim ;
    GB_nvec_nonempty_set (A, (avlen == 0) ? 0 : avdim) ;
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

    GB_IDECL (Ap, , u) ; GB_IPTR (Ap, Ap_is_32) ;
    GB_IDECL (Ai, , u) ; GB_IPTR (Ai, Ai_is_32) ;

    int64_t k ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (k = 0 ; k <= avdim ; k++)
    { 
        // Ap [k] = k * avlen ;
        GB_ISET (Ap, k, k * avlen) ;
    }

    int64_t p ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (p = 0 ; p < anz ; p++)
    { 
        // Ai [p] = p % avlen ;
        GB_ISET (Ai, p, p % avlen) ;
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

