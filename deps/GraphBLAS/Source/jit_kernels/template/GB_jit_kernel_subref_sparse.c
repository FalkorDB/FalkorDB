//------------------------------------------------------------------------------
// GB_jit_kernel_subref_sparse.c: C = A(I,J) where C and A are sparse/hyper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "include/GB_subref_method.h"

// create the qsort kernel:
#define GB_A0_t GB_Ci_TYPE
#define GB_A1_t GB_C_TYPE
#include "include/GB_qsort_1b_kernel.h"

GB_JIT_GLOBAL GB_JIT_KERNEL_SUBREF_SPARSE_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_SUBREF_SPARSE_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;

    // get C and A
    GB_Cp_DECLARE   (Cp, const) ; GB_Cp_PTR (Cp, C) ;
    GB_Ci_DECLARE_U (Ci,      ) ; GB_Ci_PTR (Ci, C) ;

    #define GB_COPY_RANGE(pC,pA,len) \
        memcpy (Cx + (pC), Ax + (pA), (len) * sizeof (GB_C_TYPE)) ;
    #define GB_COPY_ENTRY(pC,pA) Cx [pC] = Ax [pA] ;
    const GB_C_TYPE *restrict Ax = (GB_C_TYPE *) A->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #define GB_QSORT_1B(Ci,Cx,pC,clen) GB_qsort_1b_kernel (Ci+pC, Cx+pC, clen)

    // get Ap_start and Ap_end
    const GB_Ap_TYPE *restrict Ap_start = Ap_start_input ;
    const GB_Ap_TYPE *restrict Ap_end   = Ap_end_input ;

    // get I
    const GB_I_TYPE *restrict I = I_input ;

    // get I inverse lists
    const GB_IHEAD_TYPE *restrict Ihead = Ihead_input ;
    const GB_IHEAD_TYPE *restrict Inext = Inext_input ;

    #ifndef GB_Ai_IS_32
    #define GB_Ai_IS_32 (GB_Ai_BITS == 32)
    #endif

    #define GB_PHASE_2_OF_2
    #include "template/GB_subref_template.c"
    return (GrB_SUCCESS) ;
}

