//------------------------------------------------------------------------------
// GB_jit_kernel_unjumble.c: sort the vectors of a sparse/hyper matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "include/GB_qsort_1b_kernel.h"

GB_JIT_GLOBAL GB_JIT_KERNEL_UNJUMBLE_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_UNJUMBLE_PROTO (GB_jit_kernel)
{
    // get A
    const int64_t *restrict Ap = A->p ;
    int64_t *restrict Ai = A->i ;
    GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    // sort its vectors
    #define GB_QSORT GB_qsort_1b_kernel (Ai+pA_start, Ax+pA_start, aknz)
    #include "template/GB_unjumbled_template.c"
    return (GrB_SUCCESS) ;
}

