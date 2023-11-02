//------------------------------------------------------------------------------
// GB_add_bitmap_noM: C=A+B, C bitmap, A or B bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap.
// A or B is bitmap (or both).  Neither A nor B are full.

{

    //      ------------------------------------------
    //      C       =           A       +       B
    //      ------------------------------------------
    //      bitmap  .           sparse          bitmap
    //      bitmap  .           bitmap          sparse
    //      bitmap  .           bitmap          bitmap

    ASSERT (A_is_bitmap || B_is_bitmap) ;
    ASSERT (!A_is_full) ;
    ASSERT (!B_is_full) ;

    #ifdef GB_JIT_KERNEL
    {
        #if (GB_A_IS_BITMAP && GB_B_IS_BITMAP)
        {
            // A and B are both bitmap
            #include "GB_add_bitmap_noM_21.c"
        }
        #elif (GB_A_IS_BITMAP)
        {
            // A is bitmap, B is sparse/hyper
            #include "GB_add_bitmap_noM_22.c"
        }
        #else
        {
            // A is sparse/hyper, B is bitmap
            #include "GB_add_bitmap_noM_23.c"
        }
        #endif
    }
    #else
    {
        if (A_is_bitmap && B_is_bitmap)
        { 
            // A and B are both bitmap
            #include "GB_add_bitmap_noM_21.c"
        }
        else if (A_is_bitmap)
        { 
            // A is bitmap, B is sparse/hyper
            #include "GB_add_bitmap_noM_22.c"
        }
        else
        { 
            // A is sparse/hyper, B is bitmap
            #include "GB_add_bitmap_noM_23.c"
        }
    }
    #endif
}

