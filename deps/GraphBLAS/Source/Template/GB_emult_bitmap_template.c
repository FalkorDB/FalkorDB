//------------------------------------------------------------------------------
// GB_emult_bitmap_template: C = A.*B, C<M>=A.*B, and C<!M>=A.*B, C bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap.  A and B are bitmap or full.  M depends on the method

{

    //--------------------------------------------------------------------------
    // get C, A, and B
    //--------------------------------------------------------------------------

    const int8_t  *restrict Ab = A->b ;
    const int8_t  *restrict Bb = B->b ;
    const int64_t vlen = A->vlen ;

    ASSERT (GB_IS_BITMAP (A) || GB_IS_FULL (A)) ;
    ASSERT (GB_IS_BITMAP (B) || GB_IS_FULL (B)) ;

    #ifdef GB_JIT_KERNEL
    #define A_iso GB_A_ISO
    #define B_iso GB_B_ISO
    #else
    const bool A_iso = A->iso ;
    const bool B_iso = B->iso ;
    #endif

    int8_t *restrict Cb = C->b ;
    GB_C_NHELD (cnz) ;      // const int64_t cnz = GB_nnz_held (C) ;

    #ifdef GB_ISO_EMULT
    ASSERT (C->iso) ;
    #else
    ASSERT (!C->iso) ;
    ASSERT (!(A_iso && B_iso)) ;    // one of A or B can be iso, but not both
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *) B->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #endif

    #ifdef GB_JIT_KERNEL
    #define Mask_comp   GB_MASK_COMP
    #define Mask_struct GB_MASK_STRUCT
    #endif

    //--------------------------------------------------------------------------
    // C=A.*B, C<M>=A.*B, or C<!M>=A.*B: C is bitmap
    //--------------------------------------------------------------------------

    // TODO modify this method so it can modify C in-place, and also use the
    // accum operator.
    int64_t cnvals = 0 ;

    #ifdef GB_JIT_KERNEL
    {
        #if GB_NO_MASK
        {
            // C=A.*B; C bitmap, M not present, A and B are bitmap/full
            #include "GB_emult_bitmap_5.c"
        }
        #elif GB_M_IS_SPARSE || GB_M_IS_HYPER
        {
            // C<!M>=A.*B; C bitmap, M sparse/hyper, A and B are bitmap/full
            #include "GB_emult_bitmap_6.c"
        }
        #else
        {
            // C<#M>=A.*B; C bitmap; M, A, and B are all bitmap/full
            #include "GB_emult_bitmap_7.c"
        }
        #endif
    }
    #else
    {
        if (M == NULL)
        { 
            // C=A.*B; C bitmap, M not present, A and B are bitmap/full
            #include "GB_emult_bitmap_5.c"
        }
        else if (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M))
        { 
            // C<!M>=A.*B; C bitmap, M sparse/hyper, A and B are bitmap/full
            #include "GB_emult_bitmap_6.c"
        }
        else
        { 
            // C<#M>=A.*B; C bitmap; M, A, and B are all bitmap/full
            #include "GB_emult_bitmap_7.c"
        }
    }
    #endif

    C->nvals = cnvals ;
}

