//------------------------------------------------------------------------------
// GB_ew: ewise kernels for each built-in binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_control.h"
#if defined (GxB_NO_UINT8)
#define GB_TYPE_ENABLED 0
#else
#define GB_TYPE_ENABLED 1
#endif

#if GB_TYPE_ENABLED
#include "GB.h"
#include "emult/GB_emult.h"
#include "assign/GB_bitmap_assign_methods.h"
#include "FactoryKernels/GB_ew__include.h"

// operator:
#define GB_BINOP(z,x,y,i,j) z = y
#define GB_Z_TYPE uint8_t
#define GB_X_TYPE uint8_t
#define GB_Y_TYPE uint8_t
#define GB_OP_IS_SECOND 1

// A matrix:
#define GB_A_TYPE uint8_t
#define GB_A2TYPE void
#define GB_DECLAREA(aij) uint8_t aij
#define GB_GETA(aij,Ax,pA,A_iso)

// B matrix:
#define GB_B_TYPE uint8_t
#define GB_B2TYPE uint8_t
#define GB_DECLAREB(bij) uint8_t bij
#define GB_GETB(bij,Bx,pB,B_iso) bij = Bx [(B_iso) ? 0 : (pB)]

// C matrix:
#define GB_C_TYPE uint8_t

#define GB_Cp_IS_32 Cp_is_32

// disable this operator and use the generic case if these conditions hold
#if (defined(GxB_NO_SECOND) || defined(GxB_NO_UINT8) || defined(GxB_NO_SECOND_UINT8))
#define GB_DISABLE 1
#else
#define GB_DISABLE 0
#endif

#include "ewise/include/GB_ewise_shared_definitions.h"

//------------------------------------------------------------------------------
// C = A+B, all 3 matrices dense
//------------------------------------------------------------------------------

GrB_Info GB (_Cewise_fulln__second_uint8)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int nthreads
)
{ 
    #include "ewise/template/GB_ewise_fulln_template.c"
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// C = A*D, column scale with diagonal D matrix
//------------------------------------------------------------------------------

GrB_Info GB (_AxD__second_uint8)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix D,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "mxm/template/GB_colscale_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C = D*B, row scale with diagonal D matrix
//------------------------------------------------------------------------------

GrB_Info GB (_DxB__second_uint8)
(
    GrB_Matrix C,
    const GrB_Matrix D,
    const GrB_Matrix B,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "mxm/template/GB_rowscale_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseAdd: C=A+B, C<M>=A+B, C<!M>=A+B
//------------------------------------------------------------------------------

GrB_Info GB (_AaddB__second_uint8)
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
    #include "add/template/GB_add_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseUnion: C=A+B, C<M>=A+B, C<!M>=A+B
//------------------------------------------------------------------------------

GrB_Info GB (_AunionB__second_uint8)
(
    GrB_Matrix C,
    const int C_sparsity,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GB_void *alpha_scalar_in,
    const GB_void *beta_scalar_in,
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
    GB_X_TYPE alpha_scalar = (*((GB_X_TYPE *) alpha_scalar_in)) ;
    GB_Y_TYPE beta_scalar  = (*((GB_Y_TYPE *) beta_scalar_in )) ;
    #define GB_IS_EWISEUNION 1
    // for the "easy mask" condition:
    bool M_is_A = GB_all_aliased (M, A) ;
    bool M_is_B = GB_all_aliased (M, B) ;
    #include "add/template/GB_add_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseMult: C=A.*B, C<M>=A.*B, or C<M!>=A.*B where C is sparse/hyper
//------------------------------------------------------------------------------

GrB_Info GB (_AemultB_08__second_uint8)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const GB_task_struct *restrict TaskList,
    const int C_ntasks,
    const int C_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "emult/template/GB_emult_08_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseMult: C<#> = A.*B when A is sparse/hyper and B is bitmap/full
//------------------------------------------------------------------------------

GrB_Info GB (_AemultB_02__second_uint8)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const uint64_t *restrict Cp_kfirst,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "emult/template/GB_emult_02_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseMult: C<#> = A.*B when A is bitmap/full and B is sparse/hyper
//------------------------------------------------------------------------------

GrB_Info GB (_AemultB_03__second_uint8)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const uint64_t *restrict Cp_kfirst,
    const int64_t *B_ek_slicing,
    const int B_ntasks,
    const int B_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "emult/template/GB_emult_03_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseMult: C<M> = A.*B, M sparse/hyper, A and B bitmap/full
//------------------------------------------------------------------------------

GrB_Info GB (_AemultB_04__second_uint8)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const uint64_t *restrict Cp_kfirst,
    const int64_t *M_ek_slicing,
    const int M_ntasks,
    const int M_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "emult/template/GB_emult_04_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseMult: C=A.*B, C<M>=A.*B, C<!M>=A.*B where C is bitmap
//------------------------------------------------------------------------------

GrB_Info GB (_AemultB_bitmap__second_uint8)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *M_ek_slicing,
    const int M_ntasks,
    const int M_nthreads,
    const int C_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "emult/template/GB_emult_bitmap_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

#else
GB_EMPTY_PLACEHOLDER
#endif

