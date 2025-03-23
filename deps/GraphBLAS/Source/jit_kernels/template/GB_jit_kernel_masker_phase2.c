//------------------------------------------------------------------------------
// GB_jit_kernel_masker_phase2.c: R = masker(C,M,Z)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GB_JIT_GLOBAL GB_JIT_KERNEL_MASKER_PHASE2_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_MASKER_PHASE2_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;
    #define GB_PHASE_2_OF_2
    #include "template/GB_masker_template.c"
    return (GrB_SUCCESS) ;
}

