//------------------------------------------------------------------------------
// GB_jit_kernel_emult_08.c: C<#M>=A.*B, for emult_08
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GB_JIT_GLOBAL GB_JIT_KERNEL_EMULT_08_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_EMULT_08_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;
    #include "template/GB_emult_08_template.c"
    return (GrB_SUCCESS) ;
}

