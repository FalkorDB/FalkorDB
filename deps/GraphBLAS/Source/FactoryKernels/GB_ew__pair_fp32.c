//------------------------------------------------------------------------------
// GB_ew: ewise kernels for each built-in binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_control.h"
#include "GB_ewise_kernels.h"
#include "GB_ew__include.h"

// operator:
#define GB_BINOP(z,x,y,i,j) z = 1
#define GB_Z_TYPE float
#define GB_X_TYPE float
#define GB_Y_TYPE float

// A matrix:
#define GB_A_TYPE float
#define GB_A2TYPE void
#define GB_DECLAREA(aij) float aij
#define GB_GETA(aij,Ax,pA,A_iso)

// B matrix:
#define GB_B_TYPE float
#define GB_B2TYPE void
#define GB_DECLAREB(bij) float bij
#define GB_GETB(bij,Bx,pB,B_iso)

// C matrix:
#define GB_C_TYPE float

// disable this operator and use the generic case if these conditions hold
#if (defined(GxB_NO_PAIR) || defined(GxB_NO_FP32) || defined(GxB_NO_PAIR_FP32))
#define GB_DISABLE 1
#else
#define GB_DISABLE 0
#endif

#include "GB_ewise_shared_definitions.h"

//------------------------------------------------------------------------------
// C = A+B, all 3 matrices dense
//------------------------------------------------------------------------------

GrB_Info GB (_Cewise_fulln__pair_fp32)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int nthreads
)
{ 
    #include "GB_ewise_fulln_template.c"
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// eWiseAdd: C=A+B, C<M>=A+B, C<!M>=A+B
//------------------------------------------------------------------------------

GrB_Info GB (_AaddB__pair_fp32)
(
    GrB_Matrix C,
    const int C_sparsity,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const bool Ch_is_Mh,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const GB_task_struct *restrict TaskList,
    const int C_ntasks,
    const int C_nthreads,
    const int64_t *restrict M_ek_slicing,
    const int M_nthreads,
    const int M_ntasks,
    const int64_t *restrict A_ek_slicing,
    const int A_nthreads,
    const int A_ntasks,
    const int64_t *restrict B_ek_slicing,
    const int B_nthreads,
    const int B_ntasks
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #define GB_IS_EWISEUNION 0
    // for the "easy mask" condition:
    bool M_is_A = GB_all_aliased (M, A) ;
    bool M_is_B = GB_all_aliased (M, B) ;
    #include "GB_add_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseUnion: C=A+B, C<M>=A+B, C<!M>=A+B
//------------------------------------------------------------------------------

