//------------------------------------------------------------------------------
// GB_jit_kernel_add.c: C=A+B, C<#M>=A+B, for eWiseAdd
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GB_JIT_GLOBAL GB_JIT_KERNEL_ADD_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_ADD_PROTO (GB_jit_kernel)
{
    #define GB_IS_EWISEUNION 0
    #include "GB_add_template.c"
    return (GrB_SUCCESS) ;
}

