//------------------------------------------------------------------------------
// GB_aop:  assign/subassign kernels with accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C(I,J)<M> += A

#include "GB_control.h"
#if defined (GxB_NO_INT8)
#define GB_TYPE_ENABLED 0
#else
#define GB_TYPE_ENABLED 1
#endif

#if GB_TYPE_ENABLED
#include "GB.h"
#include "FactoryKernels/GB_aop__include.h"

// accum operator
#define GB_ACCUM_OP(z,x,y) z = ((x) & (y))
#define GB_Z_TYPE int8_t
#define GB_X_TYPE int8_t
#define GB_Y_TYPE int8_t
#define GB_DECLAREY(ywork) int8_t ywork
#define GB_COPY_aij_to_ywork(ywork,Ax,pA,A_iso) ywork = Ax [(A_iso) ? 0 : (pA)]

// A and C matrices
#define GB_A_TYPE int8_t
#define GB_C_TYPE int8_t
#define GB_DECLAREC(cwork) int8_t cwork
#define GB_COPY_aij_to_cwork(cwork,Ax,pA,A_iso) cwork = Ax [A_iso ? 0 : (pA)]
#define GB_COPY_aij_to_C(Cx,pC,Ax,pA,A_iso,cwork,C_iso) Cx [pC] = (A_iso) ? cwork : Ax [pA]
#define GB_COPY_cwork_to_C(Cx,pC,cwork,C_iso) Cx [pC] = cwork
#define GB_AX_MASK(Ax,pA,asize) (Ax [pA] != 0)

// C(i,j) += ywork
#define GB_ACCUMULATE_scalar(Cx,pC,ywork,C_iso) \
    GB_ACCUM_OP (Cx [pC], Cx [pC], ywork)

// C(i,j) += (ytype) A(i,j)
#define GB_ACCUMULATE_aij(Cx,pC,Ax,pA,A_iso,ywork,C_iso)    \
{                                                           \
    if (A_iso)                                              \
    {                                                       \
        GB_ACCUMULATE_scalar (Cx, pC, ywork, C_iso) ;       \
    }                                                       \
    else                                                    \
    {                                                       \
        /* A and Y have the same type here */               \
        GB_ACCUMULATE_scalar (Cx, pC, Ax [pA], C_iso) ;     \
    }                                                       \
}

// disable this operator and use the generic case if these conditions hold
#if (defined(GxB_NO_BAND) || defined(GxB_NO_INT8) || defined(GxB_NO_BAND_INT8))
#define GB_DISABLE 1
#else
#define GB_DISABLE 0
#endif

#include "assign/include/GB_assign_shared_definitions.h"

//------------------------------------------------------------------------------
// C += A, accumulate a sparse matrix into a dense matrix
//------------------------------------------------------------------------------

#undef  GB_SCALAR_ASSIGN
#define GB_SCALAR_ASSIGN 0

GrB_Info GB (_subassign_23__band_int8)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    GB_Werk Werk
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    #include "assign/template/GB_subassign_23_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C += y, accumulate a scalar into a dense matrix
//------------------------------------------------------------------------------

#undef  GB_SCALAR_ASSIGN
#define GB_SCALAR_ASSIGN 1

GrB_Info GB (_subassign_22__band_int8)
(
    GrB_Matrix C,
    const GB_void *ywork_handle
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    // get the scalar ywork for C += ywork, of type GB_Y_TYPE
    GB_Y_TYPE ywork = (*((GB_Y_TYPE *) ywork_handle)) ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    #include "assign/template/GB_subassign_22_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

#else
GB_EMPTY_PLACEHOLDER
#endif

