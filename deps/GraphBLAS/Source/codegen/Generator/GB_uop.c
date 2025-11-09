//------------------------------------------------------------------------------
// GB_uop.c:  hard-coded functions for each built-in unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_control.h"
GB_type_enabled
#if GB_TYPE_ENABLED
#include "GB.h"
#include "FactoryKernels/GB_uop__include.h"

// unary operator: z = f(x)
GB_unaryop
GB_ztype
GB_xtype

// A matrix
GB_atype
GB_declarea
GB_geta

// C matrix
GB_ctype
#define GB_Cp_IS_32 Cp_is_32

// cij = op (aij)
#define GB_APPLY_OP(pC,pA)          \
{                                   \
    /* aij = Ax [pA] */             \
    GB_DECLAREA (aij) ;             \
    GB_GETA (aij, Ax, pA, false) ;  \
    /* Cx [pC] = unaryop (aij) */   \
    GB_UNARYOP (Cx [pC], aij) ;     \
}

// disable this operator and use the generic case if these conditions hold
GB_disable

#include "omp/include/GB_kernel_shared_definitions.h"

m4_divert(if_uop_apply_enabled)
//------------------------------------------------------------------------------
// Cx = op (cast (Ax)): apply a unary operator
//------------------------------------------------------------------------------

GrB_Info GB (_uop_apply)
(
    GB_void *Cx_out,            // Cx and Ax may be aliased
    const GB_void *Ax_in,       // A is always non-iso for this kernel
    const int8_t *restrict Ab,  // A->b if A is bitmap
    int64_t anz,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "apply/factory/GB_apply_unop_template.c"
    return (GrB_SUCCESS) ;
    #endif
}
m4_divert(0)

//------------------------------------------------------------------------------
// C = op (cast (A')): transpose, typecast, and apply a unary operator
//------------------------------------------------------------------------------

GrB_Info GB (_uop_tran)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    void **Workspaces,
    const int64_t *restrict A_slice,
    int nworkspaces,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    bool Cp_is_32 = C->p_is_32 ;
    #include "transpose/template/GB_transpose_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

#else
GB_EMPTY_PLACEHOLDER
#endif

