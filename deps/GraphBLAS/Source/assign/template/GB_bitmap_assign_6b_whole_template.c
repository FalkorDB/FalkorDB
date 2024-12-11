//------------------------------------------------------------------------------
// GB_bitmap_assign_6b_whole_template: C bitmap, no M, no accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_FREE_ALL_FOR_BITMAP

{ 

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C_A_SCALAR_FOR_BITMAP

    //--------------------------------------------------------------------------
    // C = A, where C is bitmap and A is sparse/hyper
    //--------------------------------------------------------------------------

    GB_memset (Cb, 0, cnzmax, nthreads_max) ;
    cnvals = 0 ;
    #define GB_AIJ_WORK(pC,pA)                              \
    {                                                       \
        /* Cx [pC] = Ax [pA] */                             \
        GB_COPY_aij_to_C (Cx,pC,Ax,pA,A_iso,cwork,C_iso) ;  \
        Cb [pC] = 1 ;                                       \
    }
    #include "template/GB_bitmap_assign_A_whole_template.c"
    GB_A_NVALS (anz) ;
    C->nvals = anz ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    ASSERT_MATRIX_OK (C, "final C bitmap assign_6b_whole", GB0) ;
    return (GrB_SUCCESS) ;
}

