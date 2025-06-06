//------------------------------------------------------------------------------
// GB_jit_kernel_subassign_12.c: C(I,J)<M or !M,repl> += A ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GB_JIT_GLOBAL GB_JIT_KERNEL_SUBASSIGN_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_SUBASSIGN_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;
    GB_GET_CALLBACK (GB_free_memory) ;
    GB_GET_CALLBACK (GB_werk_pop) ;
    GB_GET_CALLBACK (GB_werk_push) ;
    GB_GET_CALLBACK (GB_hyper_hash_build) ;
    GB_GET_CALLBACK (GB_add_phase0) ;
    GB_GET_CALLBACK (GB_ewise_slice) ;
    GB_GET_CALLBACK (GB_subassign_IxJ_slice) ;
    GB_GET_CALLBACK (GB_Pending_ensure) ;

    const GB_I_TYPE *I = I_input ;
    const GB_J_TYPE *J = J_input ;

    #include "template/GB_subassign_12_template.c"
}

