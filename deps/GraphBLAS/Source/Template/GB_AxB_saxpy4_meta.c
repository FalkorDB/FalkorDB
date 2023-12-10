//------------------------------------------------------------------------------
// GB_AxB_saxpy4_meta: C+=A*B, C is full, A is sparse/hyper, B bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
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
    const bool B_iso = B->iso ;
    const int64_t bvlen = B->vlen ;
    const int64_t bvdim = B->vdim ;
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;
    ASSERT (B_is_bitmap || GB_IS_FULL (B)) ;

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ai = A->i ;
    const bool A_iso = A->iso ;
    const int64_t anvec = A->nvec ;
    const int64_t avlen = A->vlen ;
    const int64_t avdim = A->vdim ;
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
        #include "GB_AxB_saxpy4_template.c"
    }
    else
    { 
        // A is sparse/hyper, B is full, no mask
        #undef  GB_B_IS_BITMAP
        #define GB_B_IS_BITMAP 0
        #include "GB_AxB_saxpy4_template.c"
    }
    #undef GB_B_IS_BITMAP
}

