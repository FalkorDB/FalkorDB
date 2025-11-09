//------------------------------------------------------------------------------
// GB_jit_kernel_ewise_fulla.c: C += A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GB_JIT_GLOBAL GB_JIT_KERNEL_EWISE_FULLA_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_EWISE_FULLA_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;
    #include "template/GB_ewise_fulla_template.c"
    return (GrB_SUCCESS) ;
}

