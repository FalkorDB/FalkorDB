//------------------------------------------------------------------------------
// GB_jit_kernel_emult_bitmap.c: C<#M>=A.*B, for emult_bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GB_JIT_GLOBAL GB_JIT_KERNEL_EMULT_BITMAP_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_EMULT_BITMAP_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;
    GB_GET_CALLBACK (GB_bitmap_M_scatter_whole) ;

    #include "template/GB_emult_bitmap_template.c"
    return (GrB_SUCCESS) ;
}

