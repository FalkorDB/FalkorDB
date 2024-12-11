//------------------------------------------------------------------------------
// GB_jit_kernel_select_phase1:  select phase 1 JIT kernel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GB_JIT_GLOBAL GB_JIT_KERNEL_SELECT_PHASE1_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_SELECT_PHASE1_PROTO (GB_jit_kernel)
{
    // get callback functions
    GB_GET_CALLBACK (GB_ek_slice_merge1) ;

    #if GB_DEPENDS_ON_Y
    GB_Y_TYPE y = *((GB_Y_TYPE *) ythunk) ;
    #endif
    #include "template/GB_select_entry_phase1_template.c"
    return (GrB_SUCCESS) ;
}

