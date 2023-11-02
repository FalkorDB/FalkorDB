//------------------------------------------------------------------------------
// GB_select_bitmap_template: C=select(A,thunk) if A is bitmap or full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap.  A is bitmap or full.

{
    #ifdef GB_JIT_KERNEL
    {
        #if GB_A_IS_BITMAP
        {
            // A is bitmap
            int8_t *Ab = A->b ;
            #include "GB_select_bitmap_bitmap_template.c"
        }
        #else
        {
            // A is full
            #include "GB_select_bitmap_full_template.c"
        }
        #endif
    }
    #else
    {
        int8_t *Ab = A->b ;
        if (Ab != NULL)
        { 
            // A is bitmap
            #include "GB_select_bitmap_bitmap_template.c"
        }
        else
        { 
            // A is full
            #include "GB_select_bitmap_full_template.c"
        }
    }
    #endif
}

#undef GB_TRIL_SELECTOR
#undef GB_TRIU_SELECTOR
#undef GB_DIAG_SELECTOR
#undef GB_OFFDIAG_SELECTOR
#undef GB_ROWINDEX_SELECTOR
#undef GB_COLINDEX_SELECTOR
#undef GB_COLLE_SELECTOR
#undef GB_COLGT_SELECTOR
#undef GB_ROWLE_SELECTOR
#undef GB_ROWGT_SELECTOR

