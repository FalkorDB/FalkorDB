//------------------------------------------------------------------------------
// GB_jit_kernel_emult_03.c: C<#M>=A.*B, for emult_03
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GB_JIT_GLOBAL GB_JIT_KERNEL_EMULT_03_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_EMULT_03_PROTO (GB_jit_kernel)
{
    #include "GB_emult_03_template.c"
    return (GrB_SUCCESS) ;
}

