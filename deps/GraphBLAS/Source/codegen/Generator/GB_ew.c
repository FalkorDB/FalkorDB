//------------------------------------------------------------------------------
// GB_ew: ewise kernels for each built-in binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_control.h"
GB_type_enabled
#if GB_TYPE_ENABLED
#include "GB.h"
#include "emult/GB_emult.h"
#include "assign/GB_bitmap_assign_methods.h"
#include "FactoryKernels/GB_ew__include.h"

// operator:
GB_binaryop
GB_ztype
GB_xtype
GB_ytype
GB_op_is_second

// A matrix:
GB_atype
GB_a2type
GB_declarea
GB_geta

// B matrix:
GB_btype
GB_b2type
GB_declareb
GB_getb

// C matrix:
GB_ctype
GB_copy_a_to_c
GB_copy_b_to_c
GB_ctype_is_atype
GB_ctype_is_btype
#define GB_Cp_IS_32 Cp_is_32

// disable this operator and use the generic case if these conditions hold
GB_disable

#include "ewise/include/GB_ewise_shared_definitions.h"

m4_divert(if_is_binop_subset)
//------------------------------------------------------------------------------
// C += A+B, all 3 matrices dense
//------------------------------------------------------------------------------

// The op must be MIN, MAX, PLUS, MINUS, RMINUS, TIMES, DIV, or RDIV.

GrB_Info GB (_Cewise_fulla)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int nthreads
)
{ 
    bool A_is_B = GB_all_aliased (A, B) ;
    #include "ewise/template/GB_ewise_fulla_template.c"
    return (GrB_SUCCESS) ;
}
m4_divert(0)

//------------------------------------------------------------------------------
// C = A+B, all 3 matrices dense
//------------------------------------------------------------------------------

GrB_Info GB (_Cewise_fulln)
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

m4_divert(if_binop_is_semiring_multiplier)
//------------------------------------------------------------------------------
// C = A*D, column scale with diagonal D matrix
//------------------------------------------------------------------------------

GrB_Info GB (_AxD)
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

GrB_Info GB (_DxB)
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
m4_divert(0)

//------------------------------------------------------------------------------
// eWiseAdd: C=A+B, C<M>=A+B, C<!M>=A+B
//------------------------------------------------------------------------------

GrB_Info GB (_AaddB)
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

m4_divert(if_binop_emult_is_enabled)
GrB_Info GB (_AunionB)
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

GrB_Info GB (_AemultB_08)
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

GrB_Info GB (_AemultB_02)
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

m4_divert(if_binop_is_non_commutative)
//------------------------------------------------------------------------------
// eWiseMult: C<#> = A.*B when A is bitmap/full and B is sparse/hyper
//------------------------------------------------------------------------------

GrB_Info GB (_AemultB_03)
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
m4_divert(if_binop_emult_is_enabled)

//------------------------------------------------------------------------------
// eWiseMult: C<M> = A.*B, M sparse/hyper, A and B bitmap/full
//------------------------------------------------------------------------------

GrB_Info GB (_AemultB_04)
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

GrB_Info GB (_AemultB_bitmap)
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

m4_divert(if_binop_bind_is_enabled)
//------------------------------------------------------------------------------
// Cx = op (x,Bx):  apply a binary operator to a matrix with scalar bind1st
//------------------------------------------------------------------------------

GrB_Info GB (_bind1st)
(
    GB_void *Cx_output,         // Cx and Bx may be aliased
    const GB_void *x_input,
    const GB_void *Bx_input,
    const int8_t *restrict Bb,
    int64_t bnz,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "apply/template/GB_apply_bind1st_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// Cx = op (Ax,y):  apply a binary operator to a matrix with scalar bind2nd
//------------------------------------------------------------------------------

GrB_Info GB (_bind2nd)
(
    GB_void *Cx_output,         // Cx and Ax may be aliased
    const GB_void *Ax_input,
    const GB_void *y_input,
    const int8_t *restrict Ab,
    int64_t anz,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "apply/template/GB_apply_bind2nd_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C = op (x, A'): transpose and apply a binary operator
//------------------------------------------------------------------------------

// cij = op (x, aij)
#undef  GB_APPLY_OP
#define GB_APPLY_OP(pC,pA)              \
{                                       \
    GB_DECLAREB (aij) ;                 \
    GB_GETB (aij, Ax, pA, false) ;      \
    GB_EWISEOP (Cx, pC, x, aij, 0, 0) ; \
}

GrB_Info GB (_bind1st_tran)
(
    GrB_Matrix C,
    const GB_void *x_input,
    const GrB_Matrix A,
    void **Workspaces,
    const int64_t *restrict A_slice,
    int nworkspaces,
    int nthreads
)
{ 
    #define GB_BIND_1ST
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    GB_X_TYPE x = (*((const GB_X_TYPE *) x_input)) ;
    bool Cp_is_32 = C->p_is_32 ;
    #include "transpose/template/GB_transpose_template.c"
    return (GrB_SUCCESS) ;
    #endif
    #undef GB_BIND_1ST
}

//------------------------------------------------------------------------------
// C = op (A', y): transpose and apply a binary operator
//------------------------------------------------------------------------------

// cij = op (aij, y)
#undef  GB_APPLY_OP
#define GB_APPLY_OP(pC,pA)              \
{                                       \
    GB_DECLAREA (aij) ;                 \
    GB_GETA (aij, Ax, pA, false) ;      \
    GB_EWISEOP (Cx, pC, aij, y, 0, 0) ; \
}

GrB_Info GB (_bind2nd_tran)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GB_void *y_input,
    void **Workspaces,
    const int64_t *restrict A_slice,
    int nworkspaces,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    GB_Y_TYPE y = (*((const GB_Y_TYPE *) y_input)) ;
    bool Cp_is_32 = C->p_is_32 ;
    #include "transpose/template/GB_transpose_template.c"
    return (GrB_SUCCESS) ;
    #endif
}
m4_divert(0)

#else
GB_EMPTY_PLACEHOLDER
#endif

