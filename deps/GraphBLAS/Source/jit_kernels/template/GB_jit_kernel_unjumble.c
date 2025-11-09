//------------------------------------------------------------------------------
// GB_jit_kernel_unjumble.c: sort the vectors of a sparse/hyper matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// create the qsort kernel:
#if GB_Ai_IS_32
#define GB_A0_t uint32_t
#else
#define GB_A0_t uint64_t
#endif
#define GB_A1_t GB_A_TYPE
#include "include/GB_qsort_1b_kernel.h"

GB_JIT_GLOBAL GB_JIT_KERNEL_UNJUMBLE_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_UNJUMBLE_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;
    // get A
    GB_Ap_DECLARE   (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ai_DECLARE_U (Ai,      ) ; GB_Ai_PTR (Ai, A) ;
    GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    // sort its vectors
    #define GB_QSORT GB_qsort_1b_kernel (Ai+p, Ax+p, n)
    #include "template/GB_unjumbled_template.c"
    return (GrB_SUCCESS) ;
}

