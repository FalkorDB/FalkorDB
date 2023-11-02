//------------------------------------------------------------------------------
// GB_jit_kernel_AxB_dot2n.c: JIT kernel for C<#M>=A*B dot2n method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<#M>=A*B: dot product, C is bitmap/full, dot2n method

GB_JIT_GLOBAL GB_JIT_KERNEL_AXB_DOT2N_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_AXB_DOT2N_PROTO (GB_jit_kernel)
{
    #define GB_A_NOT_TRANSPOSED
    #include "GB_AxB_dot2_meta.c"
    return (GrB_SUCCESS) ;
}

