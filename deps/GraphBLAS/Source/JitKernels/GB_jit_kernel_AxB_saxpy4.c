//------------------------------------------------------------------------------
// GB_jit_kernel_AxB_saxpy4.c: JIT kernel for C+=A*B saxpy4 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C+=A*B: C is full, saxpy4 method

#include "GB_AxB_saxpy3_template.h"

GB_JIT_GLOBAL GB_JIT_KERNEL_AXB_SAXPY4_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_AXB_SAXPY4_PROTO (GB_jit_kernel)
{
    #include "GB_AxB_saxpy4_meta.c"
    return (GrB_SUCCESS) ;
}

