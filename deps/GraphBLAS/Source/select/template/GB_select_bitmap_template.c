//------------------------------------------------------------------------------
// GB_select_bitmap_template: C=select(A,thunk) if A is bitmap or full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap.  A is bitmap or full.

{

    //--------------------------------------------------------------------------
    // get C and A
    //--------------------------------------------------------------------------

          int8_t *restrict Cb = C->b ;
    const int8_t *restrict Ab = A->b ;

    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    const int64_t avlen = A->vlen ;
    const int64_t avdim = A->vdim ;
    const int64_t anz = avlen * avdim ;
    int64_t p, cnvals = 0 ;

    //--------------------------------------------------------------------------
    // C = select (A) where A is bitmap/full and C is bitmap
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_KERNEL
    {
        #if GB_A_IS_BITMAP
        {
            // A is bitmap
            #include "template/GB_select_bitmap_bitmap_template.c"
        }
        #else
        {
            // A is full
            #include "template/GB_select_bitmap_full_template.c"
        }
        #endif
    }
    #else
    {
        if (Ab != NULL)
        { 
            // A is bitmap
            #include "template/GB_select_bitmap_bitmap_template.c"
        }
        else
        { 
            // A is full
            #include "template/GB_select_bitmap_full_template.c"
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // update the # of entries in C
    //--------------------------------------------------------------------------

    C->nvals = cnvals ;
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

