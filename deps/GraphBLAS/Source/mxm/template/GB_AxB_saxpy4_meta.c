//------------------------------------------------------------------------------
// GB_AxB_saxpy4_meta: C+=A*B, C is full, A is sparse/hyper, B bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method is only used for built-in semirings with no typecasting in
// pre-generated kernels, or JIT kernels.  It is not used for generic methods.

// The accumulator matches the semiring monoid.
// The ANY monoid and non-atomic monoids are not supported.

// C is full.
// M is not present.
// A is sparse or hypersparse.
// B is bitmap or full.

#if GB_IS_ANY_MONOID
#error "saxpy4 not defined for the ANY monoid"
#endif

{

    //--------------------------------------------------------------------------
    // get C, M, A, and B
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_FULL (C)) ;                 // C is always full
    const int64_t cvlen = C->vlen ;
    ASSERT (C->vlen == A->vlen) ;
    ASSERT (C->vdim == B->vdim) ;
    ASSERT (A->vdim == B->vlen) ;

    #if ( !GB_HAS_IDENTITY_BYTE )
    // declare the monoid identity value
    GB_DECLARE_IDENTITY_CONST (zidentity) ;
    #endif

    const int8_t *restrict Bb = B->b ;
    const int64_t bvlen = B->vlen ;
    const int64_t bvdim = B->vdim ;

    #ifdef GB_JIT_KERNEL
    #define B_is_bitmap GB_B_IS_BITMAP
    #define B_iso GB_B_ISO
    #else
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;
    const bool B_iso = B->iso ;
    #endif

    ASSERT (GB_IS_BITMAP (B) || GB_IS_FULL (B)) ;

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;
    const int64_t anvec = A->nvec ;
    const int64_t avlen = A->vlen ;
    const int64_t avdim = A->vdim ;

    #ifdef GB_JIT_KERNEL
    #define A_is_hyper  GB_A_IS_HYPER
    #define A_is_sparse GB_A_IS_SPARSE
    #define A_iso GB_A_ISO
    #else
    const bool A_is_hyper = GB_IS_HYPERSPARSE (A) ;
    const bool A_is_sparse = GB_IS_SPARSE (A) ;
    const bool A_iso = A->iso ;
    #endif

    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;

    #if !GB_A_IS_PATTERN
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    #endif
    #if !GB_B_IS_PATTERN
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *) B->x ;
    #endif
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;

    //--------------------------------------------------------------------------
    // C += A*B, no mask, A sparse/hyper, B bitmap/full
    //--------------------------------------------------------------------------

    if (B_is_bitmap)
    { 
        // A is sparse/hyper, B is bitmap, no mask
        #undef  GB_B_IS_BITMAP
        #define GB_B_IS_BITMAP 1
        #include "template/GB_AxB_saxpy4_template.c"
    }
    else
    { 
        // A is sparse/hyper, B is full, no mask
        #undef  GB_B_IS_BITMAP
        #define GB_B_IS_BITMAP 0
        #include "template/GB_AxB_saxpy4_template.c"
    }
    #undef GB_B_IS_BITMAP
}

