//------------------------------------------------------------------------------
// GB_uop.c:  hard-coded functions for each built-in unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_control.h"
#if defined (GxB_NO_FC32)
#define GB_TYPE_ENABLED 0
#else
#define GB_TYPE_ENABLED 1
#endif

#if GB_TYPE_ENABLED
#include "GB.h"
#include "FactoryKernels/GB_uop__include.h"

// unary operator: z = f(x)
#define GB_UNARYOP(z,x) z = GB_clog1pf (x)
#define GB_Z_TYPE GxB_FC32_t
#define GB_X_TYPE GxB_FC32_t

// A matrix
#define GB_A_TYPE GxB_FC32_t
#define GB_DECLAREA(aij) GxB_FC32_t aij
#define GB_GETA(aij,Ax,pA,A_iso) aij = Ax [pA]

// C matrix
#define GB_C_TYPE GxB_FC32_t
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
#if (defined(GxB_NO_LOG1P) || defined(GxB_NO_FC32))
#define GB_DISABLE 1
#else
#define GB_DISABLE 0
#endif

#include "omp/include/GB_kernel_shared_definitions.h"

//------------------------------------------------------------------------------
// Cx = op (cast (Ax)): apply a unary operator
//------------------------------------------------------------------------------

GrB_Info GB (_uop_apply__log1p_fc32_fc32)
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

//------------------------------------------------------------------------------
// C = op (cast (A')): transpose, typecast, and apply a unary operator
//------------------------------------------------------------------------------

GrB_Info GB (_uop_tran__log1p_fc32_fc32)
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

