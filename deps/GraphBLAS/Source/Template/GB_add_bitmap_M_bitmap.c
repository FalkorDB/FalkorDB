//------------------------------------------------------------------------------
// GB_add_bitmap_M_bitmap: C<#M>=A+B, C bitmap, M bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap.
// M is bitmap or full, complemented or not, and either value or structural.
// A and B have any sparsity format but at least one is bitmap or full.

{

    //----------------------------------------------------------------------
    // C is bitmap; M is bitmap or full
    //----------------------------------------------------------------------

    //      ------------------------------------------
    //      C      <M> =        A       +       B
    //      ------------------------------------------
    //      bitmap  bitmap      sparse          bitmap
    //      bitmap  bitmap      sparse          full  
    //      bitmap  bitmap      bitmap          sparse
    //      bitmap  bitmap      bitmap          bitmap
    //      bitmap  bitmap      bitmap          full  
    //      bitmap  bitmap      full            sparse
    //      bitmap  bitmap      full            bitmap
    //      bitmap  bitmap      full            full  

    //      ------------------------------------------
    //      C      <M> =        A       +       B
    //      ------------------------------------------
    //      bitmap  full        sparse          bitmap
    //      bitmap  full        sparse          full  
    //      bitmap  full        bitmap          sparse
    //      bitmap  full        bitmap          bitmap
    //      bitmap  full        bitmap          full  
    //      bitmap  full        full            sparse
    //      bitmap  full        full            bitmap
    //      bitmap  full        full            full  

    //      ------------------------------------------
    //      C     <!M> =        A       +       B
    //      ------------------------------------------
    //      bitmap  bitmap      sparse          sparse
    //      bitmap  bitmap      sparse          bitmap
    //      bitmap  bitmap      sparse          full  
    //      bitmap  bitmap      bitmap          sparse
    //      bitmap  bitmap      bitmap          bitmap
    //      bitmap  bitmap      bitmap          full  
    //      bitmap  bitmap      full            sparse
    //      bitmap  bitmap      full            bitmap
    //      bitmap  bitmap      full            full  

    //      ------------------------------------------
    //      C     <!M> =        A       +       B
    //      ------------------------------------------
    //      bitmap  full        sparse          sparse
    //      bitmap  full        sparse          bitmap
    //      bitmap  full        sparse          full  
    //      bitmap  full        bitmap          sparse
    //      bitmap  full        bitmap          bitmap
    //      bitmap  full        bitmap          full  
    //      bitmap  full        full            sparse
    //      bitmap  full        full            bitmap
    //      bitmap  full        full            full  


    ASSERT (M_is_bitmap || M_is_full) ;
    ASSERT (A_is_bitmap || A_is_full || B_is_bitmap || B_is_full) ;

    #undef  GB_GET_MIJ     
    #define GB_GET_MIJ(p)                                       \
        bool mij = GBB_M (Mb, p) && GB_MCAST (Mx, p, msize) ;   \
        if (Mask_comp) mij = !mij ;

    #ifdef GB_JIT_KERNEL
    {
        #if (GB_A_IS_BITMAP || GB_A_IS_FULL) && (GB_B_IS_BITMAP || GB_B_IS_FULL)
        {
            // A and B are both bitmap/full
            #include "GB_add_bitmap_M_bitmap_27.c"
        }
        #elif (GB_A_IS_BITMAP || GB_A_IS_FULL)
        {
            // A is bitmap/full, B is sparse/hyper
            #include "GB_add_bitmap_M_bitmap_28.c"
        }
        #else
        {
            // A is sparse/hyper, B is bitmap/full
            #include "GB_add_bitmap_M_bitmap_29.c"
        }
        #endif
    }
    #else
    { 
        if ((A_is_bitmap || A_is_full) && (B_is_bitmap || B_is_full))
        { 
            // A and B are both bitmap/full
            #include "GB_add_bitmap_M_bitmap_27.c"
        }
        else if (A_is_bitmap || A_is_full)
        { 
            // A is bitmap/full, B is sparse/hyper
            #include "GB_add_bitmap_M_bitmap_28.c"
        }
        else
        { 
            // A is sparse/hyper, B is bitmap/full
            #include "GB_add_bitmap_M_bitmap_29.c"
        }
    }
    #endif
}

