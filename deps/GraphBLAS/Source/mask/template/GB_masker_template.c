//------------------------------------------------------------------------------
// GB_masker_template:  R = masker (C, M, Z)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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
    // get inputs
    //--------------------------------------------------------------------------

    int taskid ;

    GB_Cp_DECLARE (Cp, const) ; GB_Cp_PTR (Cp, C) ;
    GB_Ci_DECLARE (Ci, const) ; GB_Ci_PTR (Ci, C) ;
    const int64_t vlen = C->vlen ;
    #ifndef GB_JIT_KERNEL
    const bool Ci_is_32 = C->i_is_32 ;
    #define GB_Ci_IS_32 Ci_is_32
    #endif

    GB_Zp_DECLARE (Zp, const) ; GB_Zp_PTR (Zp, Z) ;
    GB_Zi_DECLARE (Zi, const) ; GB_Zi_PTR (Zi, Z) ;
    const int8_t *restrict Zb = Z->b ;
    #ifndef GB_JIT_KERNEL
    const bool Z_is_bitmap = GB_IS_BITMAP (Z) ;
    const bool Z_is_full = GB_IS_FULL (Z) ;
    const bool Zi_is_32 = Z->i_is_32 ;
    #define GB_Zi_IS_32 Zi_is_32
    #endif

    GB_Mp_DECLARE (Mp, const) ; GB_Mp_PTR (Mp, M) ;
    GB_Mi_DECLARE (Mi, const) ; GB_Mi_PTR (Mi, M) ;
    const int8_t *restrict Mb = NULL ;
    const GB_M_TYPE *restrict Mx = NULL ;
    #ifndef GB_JIT_KERNEL
    const bool Mi_is_32 = M->i_is_32 ;
    #define GB_Mi_IS_32 Mi_is_32
    #endif
    size_t msize = 0 ;
    if (M != NULL)
    { 
        Mb = M->b ;
        Mx = (GB_M_TYPE *) (GB_MASK_STRUCT ? NULL : (M->x)) ;
        msize = M->type->size ;
    }

    #if defined ( GB_PHASE_2_OF_2 )

        // phase 2
        #ifndef GB_ISO_MASKER
        #ifndef GB_JIT_KERNEL
        const bool Z_iso = Z->iso ;
        const bool C_iso = C->iso ;
        #endif
        const GB_R_TYPE *restrict Cx = (GB_R_TYPE *) C->x ;
        const GB_R_TYPE *restrict Zx = (GB_R_TYPE *) Z->x ;
              GB_R_TYPE *restrict Rx = (GB_R_TYPE *) R->x ;
        size_t rsize = R->type->size ;
        #endif
        GB_Rp_DECLARE (Rp, const) ; GB_Rp_PTR (Rp, R) ;
        GB_Rh_DECLARE (Rh, const) ; GB_Rh_PTR (Rh, R) ;
        GB_Ri_DECLARE (Ri,      ) ; GB_Ri_PTR (Ri, R) ;
        int8_t *restrict Rb = R->b ;

    #else

        // phase 1
        #ifdef GB_JIT_KERNEL
              GB_Rp_TYPE *Rp = (      GB_Rp_TYPE *) Rp_parameter ;
        const GB_Rj_TYPE *Rh = (const GB_Rj_TYPE *) Rh_parameter ;
        #else
        GB_IDECL (Rp,      , u) ; GB_IPTR (Rp, Rp_is_32) ;  // OK
        GB_IDECL (Rh, const, u) ; GB_IPTR (Rh, Rj_is_32) ;  // OK
        #endif

    #endif

    //--------------------------------------------------------------------------
    // C<#M>=Z, returnng the result in R
    //--------------------------------------------------------------------------

    #if defined ( GB_PHASE_1_OF_2 )
    { 
        // phase1: R is always sparse or hypersparse
        #include "template/GB_sparse_masker_template.c"
    }
    #else
    {
        // phase2
        if (GB_R_IS_SPARSE || GB_R_IS_HYPER)
        { 
            // R is sparse or hypersparse (phase1 and phase2)
            #include "template/GB_sparse_masker_template.c"
        }
        else // R is bitmap
        { 
            // R is bitmap (phase2 only)
            ASSERT (GB_R_IS_BITMAP) ;
            #include "template/GB_bitmap_masker_template.c"
        }
    }
    #endif
}

#undef GB_ISO_MASKER

