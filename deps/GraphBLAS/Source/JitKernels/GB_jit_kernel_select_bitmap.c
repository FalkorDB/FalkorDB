//------------------------------------------------------------------------------
// GB_jit_kernel_select_bitmap:  select bitmap JIT kernel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GB_JIT_GLOBAL GB_JIT_KERNEL_SELECT_BITMAP_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_SELECT_BITMAP_PROTO (GB_jit_kernel)
{
    #if GB_DEPENDS_ON_Y
    GB_Y_TYPE y = *((GB_Y_TYPE *) ythunk) ;
    #endif
    #include "GB_select_bitmap_template.c"
    return (GrB_SUCCESS) ;
}

