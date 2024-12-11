//------------------------------------------------------------------------------
// GB_jit_kernel_concat_bitmap: concatenate A into a bitmap matrix C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// cij = op (aij)
#define GB_COPY(pC,pA,A_iso) GB_UNOP (Cx, pC, Ax, pA, A_iso, i, j, y)

GB_JIT_GLOBAL GB_JIT_KERNEL_CONCAT_BITMAP_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_CONCAT_BITMAP_PROTO (GB_jit_kernel)
{
    // get callback functions
    GB_GET_CALLBACK (GB_ek_slice) ;
    GB_GET_CALLBACK (GB_werk_pop) ;
    GB_GET_CALLBACK (GB_werk_push) ;

    #include "template/GB_concat_bitmap_template.c"
    return (GrB_SUCCESS) ;
}

