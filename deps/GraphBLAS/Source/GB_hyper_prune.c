//------------------------------------------------------------------------------
// GB_hyper_prune: remove empty vectors from a hypersparse Ap, Ah list
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed.  Only one variant possible.

// Removes empty vectors from a hypersparse list.  On input, *Ap and *Ah are
// assumed to be NULL.  The input arrays Ap_old and Ah_old are not modified,
// and thus can be shallow content from another matrix.  New hyperlists Ap and
// Ah are allocated, for nvec vectors, all nonempty.

#include "GB.h"

GrB_Info GB_hyper_prune
(
    // output, not allocated on input:
    int64_t *restrict *p_Ap, size_t *p_Ap_size,      // size plen+1
    int64_t *restrict *p_Ah, size_t *p_Ah_size,      // size plen
    int64_t *p_nvec,                // # of vectors, all nonempty
    int64_t *p_plen,                // size of Ap and Ah
    // input, not modified
    const int64_t *Ap_old,          // size nvec_old+1
    const int64_t *Ah_old,          // size nvec_old
    const int64_t nvec_old,         // original number of vectors
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (p_Ap != NULL) ;
    ASSERT (p_Ah != NULL) ;
    ASSERT (p_nvec != NULL) ;
    ASSERT (Ap_old != NULL) ;
    ASSERT (Ah_old != NULL) ;
    ASSERT (nvec_old >= 0) ;
    (*p_Ap) = NULL ;    (*p_Ap_size) = 0 ;
    (*p_Ah) = NULL ;    (*p_Ah_size) = 0 ;
    (*p_nvec) = -1 ;

    int64_t *restrict W  = NULL ; size_t W_size  = 0 ;
    int64_t *restrict Ap = NULL ; size_t Ap_size = 0 ;
    int64_t *restrict Ah = NULL ; size_t Ah_size = 0 ;

    //--------------------------------------------------------------------------
    // determine the # of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (nvec_old, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    W = GB_MALLOC_WORK (nvec_old+1, int64_t, &W_size) ;
    if (W == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // count the # of nonempty vectors
    //--------------------------------------------------------------------------

    int64_t k ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (k = 0 ; k < nvec_old ; k++)
    { 
        // W [k] = 1 if the kth vector is nonempty; 0 if empty
        W [k] = (Ap_old [k] < Ap_old [k+1]) ;
    }

    int64_t nvec ;
    GB_cumsum (W, nvec_old, &nvec, nthreads, Werk) ;

    //--------------------------------------------------------------------------
    // allocate the result
    //--------------------------------------------------------------------------

    int64_t plen = GB_IMAX (1, nvec) ;
    Ap = GB_MALLOC (plen+1, int64_t, &Ap_size) ;
    Ah = GB_MALLOC (plen  , int64_t, &Ah_size) ;
    if (Ap == NULL || Ah == NULL)
    { 
        // out of memory
        GB_FREE_WORK (&W, W_size) ;
        GB_FREE (&Ap, Ap_size) ;
        GB_FREE (&Ah, Ah_size) ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // create the Ap and Ah result
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (k = 0 ; k < nvec_old ; k++)
    {
        if (Ap_old [k] < Ap_old [k+1])
        { 
            int64_t knew = W [k] ;
            Ap [knew] = Ap_old [k] ;
            Ah [knew] = Ah_old [k] ;
        }
    }

    Ap [nvec] = Ap_old [nvec_old] ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORK (&W, W_size) ;
    (*p_Ap) = Ap ; (*p_Ap_size) = Ap_size ;
    (*p_Ah) = Ah ; (*p_Ah_size) = Ah_size ;
    (*p_nvec) = nvec ;
    (*p_plen) = plen ;
    return (GrB_SUCCESS) ;
}

