//------------------------------------------------------------------------------
// GB_jit_kernel_AxB_dot2.c: JIT kernel for C<#M>=A'*B dot2 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<#M>=A'*B: dot product, C is bitmap/full, dot2 method

GB_JIT_GLOBAL GB_JIT_KERNEL_AXB_DOT2_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_AXB_DOT2_PROTO (GB_jit_kernel)
{
    #include "GB_AxB_dot2_meta.c"
    return (GrB_SUCCESS) ;
}

