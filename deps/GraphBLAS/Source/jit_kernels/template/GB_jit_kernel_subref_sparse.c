//------------------------------------------------------------------------------
// GB_jit_kernel_subref_sparse.c: C = A(I,J) where C and A are sparse/hyper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "include/GB_subref_method.h"
#include "include/GB_qsort_1b_kernel.h"

GB_JIT_GLOBAL GB_JIT_KERNEL_SUBREF_SPARSE_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_SUBREF_SPARSE_PROTO (GB_jit_kernel)
{

    // get C and A
    const int64_t *restrict Cp = C->p ;
    int64_t *restrict Ci = C->i ;
    #define GB_COPY_RANGE(pC,pA,len) \
        memcpy (Cx + (pC), Ax + (pA), (len) * sizeof (GB_C_TYPE)) ;
    #define GB_COPY_ENTRY(pC,pA) Cx [pC] = Ax [pA] ;
    const GB_C_TYPE *restrict Ax = (GB_C_TYPE *) A->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #define GB_QSORT_1B(Ci,Cx,pC,clen) GB_qsort_1b_kernel (Ci+pC, Cx+pC, clen)

    // C = A(I,J) where C and A are sparse/hyper
    #define GB_PHASE_2_OF_2
    #include "template/GB_subref_template.c"

    return (GrB_SUCCESS) ;
}

