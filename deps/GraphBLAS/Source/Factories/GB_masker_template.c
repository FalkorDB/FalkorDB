//------------------------------------------------------------------------------
// GB_masker_template:  R = masker (C, M, Z)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Computes C<M>=Z or C<!M>=Z, returning the result in R.  The input matrix C
// is not modified.  Effectively, this computes R=C and then R<M>=Z or R<!M>=Z.
// If the C_replace descriptor is enabled, then C has already been cleared, and
// is an empty (but non-NULL) matrix.

// phase1: does not compute R itself, but just counts the # of entries in each
// vector of R.  Fine tasks compute the # of entries in their slice of a
// single vector of R, and the results are cumsum'd.

// phase2: computes R, using the counts computed by phase1.

// FUTURE:: add special cases for C==Z, C==M, and Z==M aliases

{

    //--------------------------------------------------------------------------
    // get C, Z, M, and R
    //--------------------------------------------------------------------------

    int taskid ;

    const int64_t *restrict Cp = C->p ;
    const int64_t *restrict Ch = C->h ;
    const int8_t  *restrict Cb = C->b ;
    const int64_t *restrict Ci = C->i ;
    const int64_t vlen = C->vlen ;
    const bool C_is_hyper = GB_IS_HYPERSPARSE (C) ;
    const bool C_is_sparse = GB_IS_SPARSE (C) ;
    const bool C_is_bitmap = GB_IS_BITMAP (C) ;
    const bool C_is_full = GB_IS_FULL (C) ;
    int C_nthreads, C_ntasks ;

    const int64_t *restrict Zp = Z->p ;
    const int64_t *restrict Zh = Z->h ;
    const int8_t  *restrict Zb = Z->b ;
    const int64_t *restrict Zi = Z->i ;
    const bool Z_is_hyper = GB_IS_HYPERSPARSE (Z) ;
    const bool Z_is_sparse = GB_IS_SPARSE (Z) ;
    const bool Z_is_bitmap = GB_IS_BITMAP (Z) ;
    const bool Z_is_full = GB_IS_FULL (Z) ;
    int Z_nthreads, Z_ntasks ;

    const int64_t *restrict Mp = NULL ;
    const int64_t *restrict Mh = NULL ;
    const int8_t  *restrict Mb = NULL ;
    const int64_t *restrict Mi = NULL ;
    const GB_M_TYPE *restrict Mx = NULL ;
    const bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
    const bool M_is_sparse = GB_IS_SPARSE (M) ;
    const bool M_is_bitmap = GB_IS_BITMAP (M) ;
    const bool M_is_full = GB_IS_FULL (M) ;
    const bool M_is_sparse_or_hyper = M_is_sparse || M_is_hyper ;
    int M_nthreads, M_ntasks ;
    size_t msize = 0 ;
    if (M != NULL)
    { 
        Mp = M->p ;
        Mh = M->h ;
        Mb = M->b ;
        Mi = M->i ;
        Mx = (GB_M_TYPE *) (Mask_struct ? NULL : (M->x)) ;
        msize = M->type->size ;
    }

    #if defined ( GB_PHASE_2_OF_2 )
    const bool Z_iso = Z->iso ;
    const bool C_iso = C->iso ;
    #ifndef GB_ISO_MASKER
    const GB_void *restrict Cx = (GB_void *) C->x ;
    const GB_void *restrict Zx = (GB_void *) Z->x ;
          GB_void *restrict Rx = (GB_void *) R->x ;
    #endif
    const int64_t *restrict Rp = R->p ;
    const int64_t *restrict Rh = R->h ;
          int8_t  *restrict Rb = R->b ;
          int64_t *restrict Ri = R->i ;
    size_t rsize = R->type->size ;
    // when R is bitmap or full:
    const int64_t rnz = GB_nnz_held (R) ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    #endif

    //--------------------------------------------------------------------------
    // 
    //--------------------------------------------------------------------------

    #if defined ( GB_PHASE_1_OF_2 )
    { 
        // phase1
        #include "GB_sparse_masker_template.c"
    }
    #else
    {
        // phase2
        if (R_sparsity == GxB_SPARSE || R_sparsity == GxB_HYPERSPARSE)
        { 
            // R is sparse or hypersparse (phase1 and phase2)
            #include "GB_sparse_masker_template.c"
        }
        else // R_sparsity == GxB_BITMAP
        { 
            // R is bitmap (phase2 only)
            ASSERT (R_sparsity == GxB_BITMAP) ;
            #include "GB_bitmap_masker_template.c"
        }
    }
    #endif
}

#undef GB_ISO_MASKER

