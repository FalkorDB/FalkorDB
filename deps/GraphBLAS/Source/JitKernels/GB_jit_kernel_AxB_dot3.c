//------------------------------------------------------------------------------
// GB_jit_kernel_AxB_dot3.c: JIT kernel for C<M>=A'*B dot3 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M>=A'*B: masked dot product, C and M are both sparse or both hyper

GB_JIT_GLOBAL GB_JIT_KERNEL_AXB_DOT3_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_AXB_DOT3_PROTO (GB_jit_kernel)
{
    #include "GB_AxB_dot3_meta.c"
    return (GrB_SUCCESS) ;
}

