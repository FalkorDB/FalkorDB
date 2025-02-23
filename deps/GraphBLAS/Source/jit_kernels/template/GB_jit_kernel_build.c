//------------------------------------------------------------------------------
// GB_jit_kernel_build.c: kernel for GB_build
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GB_JIT_GLOBAL GB_JIT_KERNEL_BUILD_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_BUILD_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;
    GB_Tx_TYPE *restrict Tx = (GB_Tx_TYPE *) Tx_void ;
    const GB_Sx_TYPE *restrict Sx = (GB_Sx_TYPE *) Sx_void ;
    GB_Ti_TYPE *restrict Ti = Ti_void ;
    const GB_I_TYPE *restrict I_work = I_work_void ;
    const GB_K_TYPE *restrict K_work = K_work_void ;
    const GB_I_TYPE duplicate_entry = duplicate_entry_input ;
    #include "template/GB_bld_template.c"
    return (GrB_SUCCESS) ;
}

