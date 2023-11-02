//------------------------------------------------------------------------------
// GB_jit_kernel_AxB_saxpy3.c: saxpy3 matrix multiply for a single semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define Mask_comp   GB_MASK_COMP
#define Mask_struct GB_MASK_STRUCT
#include "GB_AxB_saxpy3_template.h"

GB_JIT_GLOBAL GB_JIT_KERNEL_AXB_SAXPY3_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_AXB_SAXPY3_PROTO (GB_jit_kernel)
{
    #ifdef GB_JIT_RUNTIME
    // get callback functions
    GB_AxB_saxpy3_cumsum_f GB_AxB_saxpy3_cumsum =
        my_callback->GB_AxB_saxpy3_cumsum_func ;
    GB_bix_alloc_f GB_bix_alloc = my_callback->GB_bix_alloc_func ;
    GB_qsort_1_f GB_qsort_1 = my_callback->GB_qsort_1_func ;
    #endif

    ASSERT (GB_IS_SPARSE (C) || GB_IS_HYPERSPARSE (C)) ;
    #include "GB_AxB_saxpy3_template.c"
    return (GrB_SUCCESS) ;
}

