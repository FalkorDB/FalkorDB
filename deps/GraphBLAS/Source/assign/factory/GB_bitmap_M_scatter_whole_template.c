//------------------------------------------------------------------------------
// GB_bitmap_M_scatter_whole_template: scatter M into/from the C bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method only handles the full assign case, where there are no I and J
// index lists.  The C and M matrices must have the same size.

// C is bitmap.  M is sparse or hypersparse, and may be jumbled.

#define GB_NO_CNVALS

{
    switch (operation)
    {

        case GB_BITMAP_M_SCATTER_PLUS_2 :       // Cb (i,j) += 2

            #undef  GB_MASK_WORK
            #define GB_MASK_WORK(pC) Cb [pC] += 2
            #include "template/GB_bitmap_assign_M_all_template.c"
            break ;

        case GB_BITMAP_M_SCATTER_MINUS_2 :      // Cb (i,j) -= 2

            #undef  GB_MASK_WORK
            #define GB_MASK_WORK(pC) Cb [pC] -= 2
            #include "template/GB_bitmap_assign_M_all_template.c"
            break ;

        case GB_BITMAP_M_SCATTER_SET_2 :        // Cb (i,j) = 2

            #undef  GB_MASK_WORK
            #define GB_MASK_WORK(pC) Cb [pC] = 2
            #include "template/GB_bitmap_assign_M_all_template.c"
            break ;

        default: ;
    }
}

#undef GB_NO_CNVALS

