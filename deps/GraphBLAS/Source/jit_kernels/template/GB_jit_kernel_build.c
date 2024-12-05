//------------------------------------------------------------------------------
// GB_jit_kernel_build.c: kernel for GB_build
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GB_JIT_GLOBAL GB_JIT_KERNEL_BUILD_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_BUILD_PROTO (GB_jit_kernel)
{
    GB_Tx_TYPE *restrict Tx = (GB_Tx_TYPE *) Tx_void ;
    const GB_Sx_TYPE *restrict Sx = (GB_Sx_TYPE *) Sx_void ;
    #include "template/GB_bld_template.c"
    return (GrB_SUCCESS) ;
}


