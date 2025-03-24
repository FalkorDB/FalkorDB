//------------------------------------------------------------------------------
// GB_jit_kernel_AxB_saxpy3.c: saxpy3 matrix multiply for a single semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// create the qsort kernel
#define GB_A0_t GB_Ci_TYPE
#include "include/GB_qsort_1_kernel.h"

#define Mask_comp   GB_MASK_COMP
#define Mask_struct GB_MASK_STRUCT
#include "include/GB_AxB_saxpy3_template.h"

GB_JIT_GLOBAL GB_JIT_KERNEL_AXB_SAXPY3_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_AXB_SAXPY3_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;
    GB_GET_CALLBACK (GB_AxB_saxpy3_cumsum) ;
    GB_GET_CALLBACK (GB_bix_alloc) ;

    ASSERT (GB_IS_SPARSE (C) || GB_IS_HYPERSPARSE (C)) ;
    #include "template/GB_AxB_saxpy3_template.c"
    return (GrB_SUCCESS) ;
}

