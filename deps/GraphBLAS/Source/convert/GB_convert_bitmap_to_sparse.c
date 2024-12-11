//------------------------------------------------------------------------------
// GB_convert_bitmap_to_sparse: convert a matrix from bitmap to sparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

#define GB_FREE_ALL                     \
{                                       \
    GB_FREE (&Cp, Cp_size) ;            \
    GB_FREE (&Ci, Ci_size) ;            \
    GB_FREE (&Cx, Cx_size) ;            \
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
    int64_t *restrict Cp = NULL ; size_t Cp_size = 0 ;
    int64_t *restrict Ci = NULL ; size_t Ci_size = 0 ;
    GB_void *restrict Cx = NULL ; size_t Cx_size = 0 ;
    Cp = GB_MALLOC (avdim+1, int64_t, &Cp_size) ; 
    Ci = GB_MALLOC (anzmax, int64_t, &Ci_size) ;
    if (Cp == NULL || Ci == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    bool Cx_shallow ;
    const bool A_iso = A->iso ;
    if (A_iso)
    { 
        // A is iso.  Remove A->x from the matrix so it is not freed by
        // GB_phybix_free.  It is not modified by GB_convert_b2s, and is
        // transplanted back into A, below.
        Cx = (GB_void *) A->x ;
        Cx_shallow = A->x_shallow ;
        Cx_size = A->x_size ;
        A->x = NULL ;
    }
    else
    {
        // A is not iso.  Allocate new space for Cx, which is filled by
        // GB_convert_b2s.
        Cx = GB_MALLOC (anzmax * asize, GB_void, &Cx_size) ;
        Cx_shallow = false ;
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

    // Cx and A->x always have the same type

    // the values are not converted if A is iso
    GB_OK (GB_convert_b2s (Cp, Ci, NULL, (A_iso) ? NULL : Cx,
        &cnvec_nonempty, A->type, A, Werk)) ;

    //--------------------------------------------------------------------------
    // free prior content of A and transplant the new content
    //--------------------------------------------------------------------------

    GB_phybix_free (A) ;         // clears A->nvals
    A->p = Cp ; A->p_size = Cp_size ; A->p_shallow = false ;
    A->i = Ci ; A->i_size = Ci_size ; A->i_shallow = false ;
    A->x = Cx ; A->x_size = Cx_size ; A->x_shallow = Cx_shallow ;
    A->iso = A_iso ;
    A->nvals = anvals ;
    ASSERT (A->nvals == Cp [avdim]) ;
    A->plen = avdim ;
    A->nvec = avdim ;
    A->nvec_nonempty = cnvec_nonempty ;
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

