//------------------------------------------------------------------------------
// GB_bitmap_M_scatter_whole: scatter M into/from the C bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method only handles the full assign case, where there are no I and J
// index lists.  The C and M matrices must have the same size.

// C is bitmap.  M is sparse or hypersparse, and may be jumbled.

#include "assign/GB_bitmap_assign_methods.h"
#define GB_GENERIC
#include "assign/include/GB_assign_shared_definitions.h"
#undef  GB_FREE_ALL
#define GB_FREE_ALL ;

GB_CALLBACK_BITMAP_M_SCATTER_WHOLE_PROTO (GB_bitmap_M_scatter_whole)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (M, "M for bitmap scatter, whole", GB0) ;
    ASSERT (GB_IS_BITMAP (C)) ;
    ASSERT (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (M_ntasks > 0) ;
    ASSERT (M_nthreads > 0) ;
    ASSERT (M_ek_slicing != NULL) ;

    //--------------------------------------------------------------------------
    // get C and M
    //--------------------------------------------------------------------------

    GB_GET_MASK
    int8_t *Cb = C->b ;
    const int64_t Cvlen = C->vlen ;
    int64_t cnvals = 0 ;    // not needed
    ASSERT ((Mx == NULL) == Mask_struct) ;

    //--------------------------------------------------------------------------
    // scatter M into the C bitmap
    //--------------------------------------------------------------------------

    if (Mx == NULL)
    { 
        #undef  GB_MCAST
        #define GB_MCAST(Mx,p,msize) 1
        #include "assign/factory/GB_bitmap_M_scatter_whole_template.c"
    }
    else
    {
        switch (msize)
        {

            default:
            case GB_1BYTE : 
            {
                uint8_t *Mx1 = (uint8_t *) Mx ;
                #undef  GB_MCAST
                #define GB_MCAST(Mx,p,msize) (Mx1 [p] != 0)
                #include "assign/factory/GB_bitmap_M_scatter_whole_template.c"
            }
            break ;

            case GB_2BYTE : 
            {
                uint16_t *Mx2 = (uint16_t *) Mx ;
                #undef  GB_MCAST
                #define GB_MCAST(Mx,p,msize) (Mx2 [p] != 0)
                #include "assign/factory/GB_bitmap_M_scatter_whole_template.c"
            }
            break ;

            case GB_4BYTE : 
            {
                uint32_t *Mx4 = (uint32_t *) Mx ;
                #undef  GB_MCAST
                #define GB_MCAST(Mx,p,msize) (Mx4 [p] != 0)
                #include "assign/factory/GB_bitmap_M_scatter_whole_template.c"
            }
            break ;

            case GB_8BYTE : 
            {
                uint64_t *Mx8 = (uint64_t *) Mx ;
                #undef  GB_MCAST
                #define GB_MCAST(Mx,p,msize) (Mx8 [p] != 0)
                #include "assign/factory/GB_bitmap_M_scatter_whole_template.c"
            }
            break ;

            case GB_16BYTE : 
            {
                uint64_t *Mx16 = (uint64_t *) Mx ;
                #undef  GB_MCAST
                #define GB_MCAST(Mx,p,msize) \
                    (Mx16 [2*(p)] != 0) || (Mx16 [2*(p)+1] != 0)
                #include "assign/factory/GB_bitmap_M_scatter_whole_template.c"
            }
            break ;
        }
    }
}

