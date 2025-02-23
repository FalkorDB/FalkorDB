//------------------------------------------------------------------------------
// GB_convert_bitmap_to_sparse: convert a matrix from bitmap to sparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

#define GB_FREE_ALL                     \
{                                       \
    GB_FREE_MEMORY (&Cp, Cp_size) ;            \
    GB_FREE_MEMORY (&Ci, Ci_size) ;            \
    GB_FREE_MEMORY (&Cx, Cx_size) ;            \
}

GrB_Info GB_convert_bitmap_to_sparse    // convert matrix from bitmap to sparse
(
    GrB_Matrix A,               // matrix to convert from bitmap to sparse
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (A, "A converting bitmap to sparse", GB0) ;
    ASSERT (!GB_IS_FULL (A)) ;
    ASSERT (GB_IS_BITMAP (A)) ;
    ASSERT (!GB_IS_SPARSE (A)) ;
    ASSERT (!GB_IS_HYPERSPARSE (A)) ;
    ASSERT (!GB_PENDING (A)) ;      // bitmap never has pending tuples
    ASSERT (!GB_JUMBLED (A)) ;      // bitmap is never jumbled
    ASSERT (!GB_ZOMBIES (A)) ;      // bitmap never has zomies

    //--------------------------------------------------------------------------
    // allocate Cp, Ci, and Cx
    //--------------------------------------------------------------------------

    const int64_t anvals = A->nvals ;
    GB_BURBLE_N (anvals, "(bitmap to sparse) ") ;
    const int64_t anzmax = GB_IMAX (anvals, 1) ;
    int64_t cnvec_nonempty ;
    const int64_t avdim = A->vdim ;
    const size_t asize = A->type->size ;
    void *Cp = NULL ; size_t Cp_size = 0 ;
    void *Ci = NULL ; size_t Ci_size = 0 ;
    void *Cx = NULL ; size_t Cx_size = 0 ;

    bool Cp_is_32, Cj_is_32, Ci_is_32 ;
    GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
        GxB_AUTO_SPARSITY, anzmax, A->vlen, avdim, Werk) ;

    size_t psize = Cp_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t isize = Ci_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;

    Cp = GB_MALLOC_MEMORY (avdim+1, psize, &Cp_size) ;
    Ci = GB_MALLOC_MEMORY (anzmax,  isize, &Ci_size) ;
    if (Cp == NULL || Ci == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    const bool A_iso = A->iso ;
    if (!A_iso)
    { 
        // A is not iso.  Allocate new space for Cx, which is filled by
        // GB_convert_b2s.
        Cx = GB_MALLOC_MEMORY (anzmax, asize, &Cx_size) ;
        if (Cx == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    //--------------------------------------------------------------------------
    // convert to sparse format (Cp, Ci, and Cx)
    //--------------------------------------------------------------------------

    // Cx and A->x always have the same type.
    // The values are not converted if A is iso (Cx is NULL).
    GB_OK (GB_convert_b2s (Cp, Ci, NULL, Cx, &cnvec_nonempty,
        Cp_is_32, Ci_is_32, false, A->type, A, Werk)) ;

    //--------------------------------------------------------------------------
    // free prior content of A and transplant the new content
    //--------------------------------------------------------------------------

    bool Cx_shallow = false ;
    if (A_iso)
    { 
        // A is iso.  Remove A->x from the matrix so it is not freed by
        // GB_phybix_free; it is transplanted back again just below.
        Cx = A->x ;
        Cx_size = A->x_size ;
        Cx_shallow = A->x_shallow ;
        A->x = NULL ;
    }

    GB_phybix_free (A) ;        // clears A->nvals
    A->p = Cp ; A->p_size = Cp_size ; A->p_shallow = false ;
    A->i = Ci ; A->i_size = Ci_size ; A->i_shallow = false ;
    A->x = Cx ; A->x_size = Cx_size ; A->x_shallow = Cx_shallow ;
    A->p_is_32 = Cp_is_32 ;
    A->j_is_32 = Cj_is_32 ;
    A->i_is_32 = Ci_is_32 ;
    A->iso = A_iso ;
    A->nvals = anvals ;
    A->plen = avdim ;
    A->nvec = avdim ;
//  A->nvec_nonempty = cnvec_nonempty ;
    GB_nvec_nonempty_set (A, cnvec_nonempty) ;
    A->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converted from to bitmap to sparse", GB0) ;
    ASSERT (GB_IS_SPARSE (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    return (GrB_SUCCESS) ;
}

