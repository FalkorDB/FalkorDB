//------------------------------------------------------------------------------
// GB_add_bitmap_template: C=A+B, C<#M>=A+B, C bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap.  The mask M can have any sparsity structure, and is efficient
// to apply (all methods are asymptotically optimal).  All cases (no M, M, !M)
// are handled.  The values of A, B, and C are not accessed if C is iso, in
// which case GB_ISO_ADD is #defined by the #including file.

{

    // TODO: the input C can be modified in-place, if it is also bitmap
    int64_t cnvals = 0 ;

    #ifdef GB_JIT_KERNEL
    {

        #if GB_NO_MASK
        {
            // M is not present.
            // A or B is bitmap (or both).  Neither A nor B are full.
            #include "GB_add_bitmap_noM.c"
        }
        #elif (GB_M_IS_SPARSE || GB_M_IS_HYPER)
        {
            // M is sparse/hyper and complemented, value/structural.
            // A and B can have any format, except at least one is bitmap/full.
            #include "GB_add_bitmap_M_sparse.c"
        }
        #else
        {
            // M is bitmap/full, complemented or not, and valued/structural.
            // A and B have any sparsity format but at least one is bitmap/full.
            #include "GB_add_bitmap_M_bitmap.c"
        }
        #endif

    }
    #else
    {

        if (M == NULL)
        { 
            // M is not present.
            // A or B is bitmap (or both).  Neither A nor B are full.
            #include "GB_add_bitmap_noM.c"
        }
        else if (M_is_sparse_or_hyper)
        { 
            // M is sparse/hyper and complemented, value/structural.
            // A and B can have any format, except at least one is bitmap/full.
            #include "GB_add_bitmap_M_sparse.c"
        }
        else
        { 
            // M is bitmap/full, complemented or not, and valued/structural.
            // A and B have any sparsity format but at least one is bitmap/full.
            #include "GB_add_bitmap_M_bitmap.c"
        }
    }
    #endif

    C->nvals = cnvals ;
}

