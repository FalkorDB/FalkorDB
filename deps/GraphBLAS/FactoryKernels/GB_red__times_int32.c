//------------------------------------------------------------------------------
// GB_red:  hard-coded functions for reductions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_control.h"
#if defined (GxB_NO_INT32)
#define GB_TYPE_ENABLED 0
#else
#define GB_TYPE_ENABLED 1
#endif

#if GB_TYPE_ENABLED
#include "GB.h"
#include "FactoryKernels/GB_red__include.h"

// reduction operator and type:
#define GB_UPDATE(z,a)  z *= a
#define GB_ADD(z,zin,a) z = zin * a
#define GB_GETA_AND_UPDATE(z,Ax,p) z *= Ax [p]

// A matrix (no typecasting to Z type here)
#define GB_A_TYPE  int32_t
#define GB_DECLAREA(aij) int32_t aij
#define GB_GETA(aij,Ax,pA,A_iso) aij = Ax [pA]

// monoid properties:
#define GB_Z_TYPE  int32_t
#define GB_DECLARE_IDENTITY(z) int32_t z = 1
#define GB_DECLARE_IDENTITY_CONST(z) const int32_t z = 1

#define GB_MONOID_IS_TERMINAL 1
#define GB_TERMINAL_CONDITION(z,zterminal) (z == 0)
#define GB_IF_TERMINAL_BREAK(z,zterminal) if (z == 0) { break ; }

// panel size
#define GB_PANEL 64

// disable this operator and use the generic case if these conditions hold
#if (defined(GxB_NO_TIMES) || defined(GxB_NO_INT32) || defined(GxB_NO_TIMES_INT32))
#define GB_DISABLE 1
#else
#define GB_DISABLE 0
#endif

#include "monoid/include/GB_monoid_shared_definitions.h"

//------------------------------------------------------------------------------
// reduce to a non-iso matrix to scalar, for monoids only
//------------------------------------------------------------------------------

GrB_Info GB (_red__times_int32)
(
    GB_Z_TYPE *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    GB_Z_TYPE z = (*result) ;
    GB_Z_TYPE *restrict W = (GB_Z_TYPE *) W_space ;
    if (A->nzombies > 0 || GB_IS_BITMAP (A))
    {
        #include "reduce/template/GB_reduce_to_scalar_template.c"
    }
    else
    {
        #include "reduce/template/GB_reduce_panel.c"
    }
    (*result) = z ;
    return (GrB_SUCCESS) ;
    #endif
}

#else
GB_EMPTY_PLACEHOLDER
#endif

