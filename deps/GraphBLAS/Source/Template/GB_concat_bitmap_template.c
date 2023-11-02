//------------------------------------------------------------------------------
// GB_concat_bitmap_template: concatenate a tile into a bitmap matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get C and the tile A
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_KERNEL
    #define A_iso GB_A_ISO
    const int64_t avdim = A->vdim ;
    const int64_t avlen = A->vlen ;
    const int64_t cvlen = C->vlen ;
    const int64_t anz = avlen * avdim ;
    #else
    const bool A_iso = A->iso ;
    #endif

    #ifndef GB_ISO_CONCAT
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #endif

    int8_t *restrict Cb = C->b ;

    //--------------------------------------------------------------------------
    // copy the tile A into C
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_KERNEL
    {
        #if GB_A_IS_FULL
            #include "GB_concat_bitmap_full.c"
        #elif GB_A_IS_BITMAP
            #include "GB_concat_bitmap_bitmap.c"
        #else
            #include "GB_concat_bitmap_sparse.c"
        #endif
    }
    #else
    {
        switch (GB_sparsity (A))
        {

            case GxB_FULL : // A is full
            {
                #include "GB_concat_bitmap_full.c"
            }
            break ;

            case GxB_BITMAP : // A is bitmap
            {
                #include "GB_concat_bitmap_bitmap.c"
            }
            break ;

            default : // A is sparse or hypersparse
            {
                #include "GB_concat_bitmap_sparse.c"
            }
            break ;
        }
    }
    #endif
}

#undef GB_C_TYPE
#undef GB_A_TYPE
#undef GB_ISO_CONCAT

