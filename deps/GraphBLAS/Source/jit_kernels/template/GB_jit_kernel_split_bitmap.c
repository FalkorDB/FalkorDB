//------------------------------------------------------------------------------
// GB_jit_kernel_split_bitmap: split bitmap A into a bitmap tile C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// cij = op (aij)
#define GB_COPY(pC,pA) GB_UNOP (Cx, pC, Ax, pA, false, i, j, y)

GB_JIT_GLOBAL GB_JIT_KERNEL_SPLIT_BITMAP_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_SPLIT_BITMAP_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;
    #include "template/GB_split_bitmap_template.c"
    return (GrB_SUCCESS) ;
}

