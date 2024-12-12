//------------------------------------------------------------------------------
// GB_hypermatrix_prune: prune empty vectors from a hypersparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// On input, A->p and A->h may be shallow.  If modified, new arrays A->p and
// A->h are created, which are not shallow.  If these arrays are not modified,
// and are shallow on input, then they remain shallow on output.

#include "GB.h"

GrB_Info GB_hypermatrix_prune
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
    ASSERT_MATRIX_OK (A, "A before hypermatrix_prune", GB0) ;

    if (!GB_IS_HYPERSPARSE (A))
    { 
        // nothing to do
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // count # of empty vectors
    //--------------------------------------------------------------------------

    if (A->nvec_nonempty < 0)
    { 
        // A->nvec_nonempty is needed to prune the hyperlist
        A->nvec_nonempty = GB_nvec_nonempty (A) ;
    }

    //--------------------------------------------------------------------------
    // prune empty vectors
    //--------------------------------------------------------------------------

    if (A->nvec_nonempty < A->nvec)     // A->nvec_nonempty used here
    {
        // create new Ap_new and Ah_new arrays, with no empty vectors
        int64_t *restrict Ap_new = NULL ; size_t Ap_new_size = 0 ;
        int64_t *restrict Ah_new = NULL ; size_t Ah_new_size = 0 ;
        int64_t nvec_new, plen_new ;
        int64_t anz = A->nvals ;
        ASSERT (anz == A->p [A->nvec]) ;
        GrB_Info info = GB_hyper_prune (&Ap_new, &Ap_new_size,
            &Ah_new, &Ah_new_size, &nvec_new, &plen_new,
            A->p, A->h, A->nvec, Werk) ;
        if (info != GrB_SUCCESS)
        { 
            // out of memory
            return (info) ;
        }
        // free the old A->p, A->h, and A->Y
        GB_phy_free (A) ;
        // A->p, A->h, A->Y are now NULL and thus not shallow
        ASSERT (!A->p_shallow) ;
        ASSERT (!A->h_shallow) ;
        ASSERT (!A->Y_shallow) ;
        // transplant the new hyperlist into A
        A->p = Ap_new ; A->p_size = Ap_new_size ;
        A->h = Ah_new ; A->h_size = Ah_new_size ;
        A->nvec = nvec_new ;
        A->plen = plen_new ;
        A->nvec_nonempty = nvec_new ;
        A->nvals = anz ;
        ASSERT (anz == A->p [A->nvec]) ;
        A->magic = GB_MAGIC ;
    }

    ASSERT_MATRIX_OK (A, "A after hypermatrix_prune", GB0) ;
    return (GrB_SUCCESS) ;
}

