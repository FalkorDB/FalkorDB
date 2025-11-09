//------------------------------------------------------------------------------
// GB_hyper_prune: prune empty vectors from a hypersparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// On input, A->p and A->h may be shallow.  If modified, new arrays A->p and
// A->h are created, which are not shallow, and any existing hyper hash is
// freed.  If these arrays are not modified, and are shallow on input, then
// they remain shallow on output.  If new A->p and A->h arrays are constructed,
// the existing A->Y hyper_hash is freed.  A->p_is_32, A->j_is_32, and
// A->i_is_32 are unchanged in all cases.

#include "GB.h"

GrB_Info GB_hyper_prune
(
    GrB_Matrix A,               // matrix to prune
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (A != NULL) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;        // pattern not accessed
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT_MATRIX_OK (A, "A before hyper_prune", GB0) ;

    if (!GB_IS_HYPERSPARSE (A))
    { 
        // nothing to do
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // count # of empty vectors and check if pruning is needed
    //--------------------------------------------------------------------------

    // A->nvec_nonempty is needed to prune the hyperlist
    int64_t nvec_nonempty = GB_nvec_nonempty_update (A) ;
    if (nvec_nonempty == A->nvec)
    { 
        // nothing to prune
        return (GrB_SUCCESS) ;
    }
    #ifdef GB_DEBUG
    int64_t nvec_save = nvec_nonempty ;
    #endif

    //--------------------------------------------------------------------------
    // prune empty vectors
    //--------------------------------------------------------------------------

    GB_Ap_DECLARE (Ap_old, const) ; GB_Ap_PTR (Ap_old, A) ;
    GB_Ah_DECLARE (Ah_old, const) ; GB_Ah_PTR (Ah_old, A) ;

    GB_Ap_DECLARE (Ap_new, ) ; size_t Ap_new_size = 0 ;
    GB_Ah_DECLARE (Ah_new, ) ; size_t Ah_new_size = 0 ;

    GB_MDECL (W, , u) ; size_t W_size = 0 ;

    int64_t nvec_old = A->nvec ;

    size_t psize = (A->p_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t jsize = (A->j_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;

    //--------------------------------------------------------------------------
    // determine the # of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (nvec_old, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    W = GB_MALLOC_MEMORY (nvec_old+1, jsize, &W_size) ;
    if (W == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }
    GB_IPTR (W, A->j_is_32) ;

    //--------------------------------------------------------------------------
    // count the # of nonempty vectors and mark their locations in W
    //--------------------------------------------------------------------------

    int64_t k ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (k = 0 ; k < nvec_old ; k++)
    { 
        // W [k] = 1 if the kth vector is nonempty; 0 if empty
        int nonempty = (GB_IGET (Ap_old, k) < GB_IGET (Ap_old, k+1)) ;
        // W [k] = nonempty ;
        GB_ISET (W, k, nonempty) ;
    }

    int64_t nvec_new ;
    GB_cumsum (W, A->j_is_32, nvec_old, &nvec_new, nthreads, Werk) ;

    //--------------------------------------------------------------------------
    // allocate the result
    //--------------------------------------------------------------------------

    int64_t plen_new = GB_IMAX (1, nvec_new) ;
    Ap_new = GB_MALLOC_MEMORY (plen_new+1, psize, &Ap_new_size) ;
    Ah_new = GB_MALLOC_MEMORY (plen_new  , jsize, &Ah_new_size) ;
    if (Ap_new == NULL || Ah_new == NULL)
    { 
        // out of memory
        GB_FREE_MEMORY (&W, W_size) ;
        GB_FREE_MEMORY (&Ap_new, Ap_new_size) ;
        GB_FREE_MEMORY (&Ah_new, Ah_new_size) ;
        return (GrB_OUT_OF_MEMORY) ;
    }
    GB_IPTR (Ap_new, A->p_is_32) ;
    GB_IPTR (Ah_new, A->j_is_32) ;

    //--------------------------------------------------------------------------
    // create the Ap_new and Ah_new result
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (k = 0 ; k < nvec_old ; k++)
    {
        uint64_t p = GB_IGET (Ap_old, k) ;
        if (p < GB_IGET (Ap_old, k+1))
        { 
            uint64_t j = GB_IGET (Ah_old, k) ;
            uint64_t knew = GB_IGET (W, k) ;
            // Ap_new [knew] = p ;
            GB_ISET (Ap_new, knew, p) ;
            // Ah_new [knew] = j ;
            GB_ISET (Ah_new, knew, j) ;
        }
    }

    // Ap_new [nvec_new] = Ap_old [nvec_old] ;
    uint64_t nvals = A->nvals ;
    ASSERT (nvals == GB_IGET (Ap_old, nvec_old)) ;
    GB_ISET (Ap_new, nvec_new, nvals) ;

    //--------------------------------------------------------------------------
    // free workspace and old matrix components, including the A->Y hyper_hash
    //--------------------------------------------------------------------------

    GB_FREE_MEMORY (&W, W_size) ;
    GB_phy_free (A) ;

    //--------------------------------------------------------------------------
    // transplant the new hyperlist into A
    //--------------------------------------------------------------------------

    A->p = Ap_new ; A->p_size = Ap_new_size ;
    A->h = Ah_new ; A->h_size = Ah_new_size ;
    A->nvec = nvec_new ;
    A->plen = plen_new ;
    ASSERT (nvec_new == nvec_save) ;
    GB_nvec_nonempty_set (A, nvec_new) ;
    A->nvals = nvals ;
    A->magic = GB_MAGIC ;

    ASSERT_MATRIX_OK (A, "A after hyper_prune", GB0) ;
    return (GrB_SUCCESS) ;
}

