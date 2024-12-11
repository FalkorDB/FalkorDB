//------------------------------------------------------------------------------
// GB_jit_kernel_bitmap_assign_1.c: C bitmap, M bitmap/full, with accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GB_JIT_GLOBAL GB_JIT_KERNEL_SUBASSIGN_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_SUBASSIGN_PROTO (GB_jit_kernel)
{
    // get callback functions
    GB_GET_CALLBACK (GB_ek_slice) ;
    GB_GET_CALLBACK (GB_free_memory) ;
    GB_GET_CALLBACK (GB_werk_pop) ;
    GB_GET_CALLBACK (GB_werk_push) ;
    GB_GET_CALLBACK (GB_subassign_IxJ_slice) ;

    #include "template/GB_bitmap_assign_1_template.c"
}

