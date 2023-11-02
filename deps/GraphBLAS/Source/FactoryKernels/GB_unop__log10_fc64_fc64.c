//------------------------------------------------------------------------------
// GB_unop:  hard-coded functions for each built-in unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_control.h"
#include "GB_unop__include.h"

// unary operator: z = f(x)
#define GB_UNARYOP(z,x) z = GB_clog10 (x)
#define GB_Z_TYPE GxB_FC64_t
#define GB_X_TYPE GxB_FC64_t

// A matrix
#define GB_A_TYPE GxB_FC64_t
#define GB_DECLAREA(aij) GxB_FC64_t aij
#define GB_GETA(aij,Ax,pA,A_iso) aij = Ax [pA]

// C matrix
#define GB_C_TYPE GxB_FC64_t

// cij = op (aij)
#define GB_APPLY_OP(pC,pA)          \
{                                   \
    /* aij = Ax [pA] */             \
    GB_DECLAREA (aij) ;             \
    GB_GETA (aij, Ax, pA, false) ;  \
    /* Cx [pC] = unop (aij) */      \
    GB_UNARYOP (Cx [pC], aij) ;     \
}

// disable this operator and use the generic case if these conditions hold
#if (defined(GxB_NO_LOG10) || defined(GxB_NO_FC64))
#define GB_DISABLE 1
#else
#define GB_DISABLE 0
#endif

#include "GB_apply_shared_definitions.h"

//------------------------------------------------------------------------------
// Cx = op (cast (Ax)): apply a unary operator
//------------------------------------------------------------------------------

GrB_Info GB (_unop_apply__log10_fc64_fc64)
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
    #include "GB_apply_unop_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C = op (cast (A')): transpose, typecast, and apply a unary operator
//------------------------------------------------------------------------------

GrB_Info GB (_unop_tran__log10_fc64_fc64)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    int64_t *restrict *Workspaces,
    const int64_t *restrict A_slice,
    int nworkspaces,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_transpose_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

