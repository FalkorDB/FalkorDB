//------------------------------------------------------------------------------
// GB_add_full_template:  C=A+B; C is full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is full.  The mask M is not present (otherwise, C would be sparse,
// hypersparse, or bitmap).  All of these methods are asymptotically optimal.

    //      ------------------------------------------
    //      C       =           A       +       B
    //      ------------------------------------------
    //      full    .           sparse          full  
    //      full    .           bitmap          full  
    //      full    .           full            sparse
    //      full    .           full            bitmap
    //      full    .           full            full  

// If C is iso and full, this phase has nothing to do.

#ifndef GB_ISO_ADD
{

    int64_t p ;
    ASSERT (M == NULL) ;
    ASSERT (A_is_full || B_is_full) ;

    #ifdef GB_JIT_KERNEL
    {
        #if (GB_A_IS_FULL && GB_B_IS_FULL)
        {
            // C, A, and B are all full
            #include "GB_add_full_30.c"
        }
        #elif (GB_A_IS_FULL)
        {
            // C and A are full; B is hypersparse, sparse, or bitmap
            #if (GB_B_IS_BITMAP)
            {
                // C and A are full; B is bitmap
                #include "GB_add_full_31.c"
            }
            #else
            {
                // C and A are full; B is sparse or hypersparse
                #include "GB_add_full_32.c"
            }
            #endif
        }
        #else
        {
            // C and B are full; A is hypersparse, sparse, or bitmap
            #if (GB_A_IS_BITMAP)
            {
                // C and B are full; A is bitmap
                #include "GB_add_full_33.c"
            }
            #else
            {
                // C and B are full; A is hypersparse or sparse
                #include "GB_add_full_34.c"
            }
            #endif
        }
        #endif
    }
    #else
    {
        if (A_is_full && B_is_full)
        { 
            // C, A, and B are all full
            #include "GB_add_full_30.c"
        }
        else if (A_is_full)
        {
            // C and A are full; B is hypersparse, sparse, or bitmap
            if (B_is_bitmap)
            { 
                // C and A are full; B is bitmap
                #include "GB_add_full_31.c"
            }
            else
            { 
                // C and A are full; B is sparse or hypersparse
                #include "GB_add_full_32.c"
            }
        }
        else
        {
            // C and B are full; A is hypersparse, sparse, or bitmap
            if (A_is_bitmap)
            { 
                // C and B are full; A is bitmap
                #include "GB_add_full_33.c"
            }
            else
            { 
                // C and B are full; A is hypersparse or sparse
                #include "GB_add_full_34.c"
            }
        }
    }
    #endif

}
#endif

