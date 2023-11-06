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
    #ifdef GB_JIT_RUNTIME
    // get callback functions
    GB_ek_slice_f GB_ek_slice = my_callback->GB_ek_slice_func ;
    GB_werk_pop_f GB_werk_pop = my_callback->GB_werk_pop_func ;
    GB_werk_push_f GB_werk_push = my_callback->GB_werk_push_func ;
    #endif

    #include "GB_concat_bitmap_template.c"
    return (GrB_SUCCESS) ;
}

