//------------------------------------------------------------------------------
// GB_bitmap_expand_to_hyper:  expand a compact bitmap C to hypersparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed.  Only one variant possible.

#define GB_FREE_ALL                 \
{                                   \
    GB_phybix_free (C) ;            \
    GB_FREE (&Cp, Cp_size) ;        \
    GB_FREE (&Ch, Ch_size) ;        \
    GB_FREE (&Ci, Ci_size) ;        \
}

#include "GB_mxm.h"

GrB_Info GB_bitmap_expand_to_hyper
(
    // input/output:
    GrB_Matrix C,
    // input
    int64_t cvlen_final,
    int64_t cvdim_final,
    GrB_Matrix A,
    GrB_Matrix B,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (C != NULL && (GB_IS_BITMAP (C) || GB_IS_FULL (C))) ;
    ASSERT (A != NULL && B != NULL) ;
    GBURBLE ("(expand bitmap/full to hyper) ") ;
    ASSERT_MATRIX_OK (C, "C to expand from bitmap/full to hyper", GB0) ;
    ASSERT_MATRIX_OK (A, "A for expand C from bitmap/full to hyper", GB0) ;
    ASSERT_MATRIX_OK (B, "B for expand C from bitmap/full to hyper", GB0) ;

    int64_t cvlen = C->vlen ;
    int64_t cvdim = C->vdim ;
    int64_t cnz = cvlen * cvdim ;
    bool A_is_hyper = GB_IS_HYPERSPARSE (A) ;
    bool B_is_hyper = GB_IS_HYPERSPARSE (B) ;

    // C is currently a subset of its final dimension, in bitmap or full form.
    // It is converted back into sparse/hypersparse form, with zombies if
    // bitmap, and expanded in size to be cvlen_final by cvdim_final (A->vdim
    // by B->vdim for C=A'*B, or A->vlen by B->vdim for C=A*B).

    //----------------------------------------------------------------------
    // allocate the sparse/hypersparse structure of the final C
    //----------------------------------------------------------------------

    int64_t *restrict Cp = NULL ; size_t Cp_size = 0 ;
    int64_t *restrict Ch = NULL ; size_t Ch_size = 0 ;
    int64_t *restrict Ci = NULL ; size_t Ci_size = 0 ;

    Cp = GB_MALLOC (cvdim+1, int64_t, &Cp_size) ;
    Ch = NULL ;
    if (B_is_hyper)
    { 
        Ch = GB_MALLOC (cvdim, int64_t, &Ch_size) ;
    }
    Ci = GB_MALLOC (cnz, int64_t, &Ci_size) ;
    if (Cp == NULL || (B_is_hyper && Ch == NULL) || Ci == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //----------------------------------------------------------------------
    // construct the hyperlist of C, if B is hypersparse
    //----------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (cvdim, chunk, nthreads_max) ;
    if (B_is_hyper)
    { 
        // C becomes hypersparse
        ASSERT (cvdim == B->nvec) ;
        GB_memcpy (Ch, B->h, cvdim * sizeof (int64_t), nthreads) ;
    }

    //----------------------------------------------------------------------
    // construct the vector pointers of C
    //----------------------------------------------------------------------

    int64_t pC ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (pC = 0 ; pC < cvdim+1 ; pC++)
    { 
        Cp [pC] = pC * cvlen ;
    }

    //----------------------------------------------------------------------
    // construct the pattern of C from its bitmap
    //----------------------------------------------------------------------

    // C(i,j) becomes a zombie if not present in the bitmap
    nthreads = GB_nthreads (cnz, chunk, nthreads_max) ;

    int8_t *restrict Cb = C->b ;
    bool C_is_bitmap = (Cb != NULL) ;
    if (C_is_bitmap)
    {
        // C is bitmap
        if (A_is_hyper)
        { 
            // only for C=A'*B
            GrB_Index *restrict Ah = (GrB_Index *) A->h ;
            ASSERT (cvlen == A->nvec) ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (pC = 0 ; pC < cnz ; pC++)
            {
                int64_t i = Ah [pC % cvlen] ;
                Ci [pC] = (Cb [pC]) ? i : GB_FLIP (i) ;
            }
        }
        else
        { 
            // for C=A'*B or C=A*B
            ASSERT (cvlen == cvlen_final) ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (pC = 0 ; pC < cnz ; pC++)
            {
                int64_t i = pC % cvlen ;
                Ci [pC] = (Cb [pC]) ? i : GB_FLIP (i) ;
            }
        }
    }
    else
    {
        // C is full
        if (A_is_hyper)
        { 
            // only for C=A'*B
            GrB_Index *restrict Ah = (GrB_Index *) A->h ;
            ASSERT (cvlen == A->nvec) ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (pC = 0 ; pC < cnz ; pC++)
            {
                int64_t i = Ah [pC % cvlen] ;
                Ci [pC] = i ;
            }
        }
        else
        { 
            // for C=A'*B or C=A*B
            ASSERT (cvlen == cvlen_final) ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (pC = 0 ; pC < cnz ; pC++)
            {
                int64_t i = pC % cvlen ;
                Ci [pC] = i ;
            }
        }
    }

    //----------------------------------------------------------------------
    // transplant the new content and finalize C
    //----------------------------------------------------------------------

    C->p = Cp ; Cp = NULL ; C->p_size = Cp_size ;
    C->h = Ch ; Ch = NULL ; C->h_size = Ch_size ;
    C->i = Ci ; Ci = NULL ; C->i_size = Ci_size ;
    C->nzombies = (C_is_bitmap) ? (cnz - C->nvals) : 0 ;
    C->vdim = cvdim_final ;
    C->vlen = cvlen_final ;
    C->nvals = cnz ;
    C->nvec = cvdim ;
    C->plen = cvdim ;
    C->nvec_nonempty = (cvlen == 0) ? 0 : cvdim ;

    // free the bitmap, if present
    GB_FREE ((&C->b), C->b_size) ;

    // C is now sparse or hypersparse
    ASSERT_MATRIX_OK (C, "C expanded from bitmap/full to hyper", GB0) ;
    ASSERT (GB_ZOMBIES_OK (C)) ;
    return (GrB_SUCCESS) ;
}

