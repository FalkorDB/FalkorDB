//------------------------------------------------------------------------------
// GB_jit_kernel_select_phase2:  select phase 2 JIT kernel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GB_JIT_GLOBAL GB_JIT_KERNEL_SELECT_PHASE2_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_SELECT_PHASE2_PROTO (GB_jit_kernel)
{
    GB_A_TYPE *restrict Cx = (GB_A_TYPE *) Cx_out ;
    #if GB_DEPENDS_ON_Y
    GB_Y_TYPE y = *((GB_Y_TYPE *) ythunk) ;
    #endif
    #include "GB_select_phase2.c"
    return (GrB_SUCCESS) ;
}

