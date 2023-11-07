//------------------------------------------------------------------------------
// GB_bitmap_M_scatter: scatter M into/from the C bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed, but variants possible for each kind of mask matrix.

#include "GB_bitmap_assign_methods.h"
#include "GB_assign_shared_definitions.h"

GB_CALLBACK_BITMAP_M_SCATTER_PROTO (GB_bitmap_M_scatter)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (M, "M for bitmap scatter", GB0) ;
    ASSERT (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M)) ;
    ASSERT (M_ntasks > 0) ;
    ASSERT (M_nthreads > 0) ;
    ASSERT (M_ek_slicing != NULL) ;

    //--------------------------------------------------------------------------
    // get C and M
    //--------------------------------------------------------------------------

    GB_GET_M
    int8_t *Cb = C->b ;
    const int64_t cvlen = C->vlen ;
    int64_t cnvals = 0 ;

    //--------------------------------------------------------------------------
    // scatter M into the C bitmap
    //--------------------------------------------------------------------------

    switch (operation)
    {

        case GB_BITMAP_M_SCATTER_PLUS_2 :       // Cb (i,j) += 2

            #undef  GB_MASK_WORK
            #define GB_MASK_WORK(pC) Cb [pC] += 2
            #include "GB_bitmap_assign_M_template.c"
            break ;

        case GB_BITMAP_M_SCATTER_MINUS_2 :      // Cb (i,j) -= 2

            #undef  GB_MASK_WORK
            #define GB_MASK_WORK(pC) Cb [pC] -= 2
            #include "GB_bitmap_assign_M_template.c"
            break ;

        case GB_BITMAP_M_SCATTER_MOD_2 :        // Cb (i,j) %= 2

            #undef  GB_MASK_WORK
            #define GB_MASK_WORK(pC) Cb [pC] %= 2
            #include "GB_bitmap_assign_M_template.c"
            break ;

        default: ;
    }
}

