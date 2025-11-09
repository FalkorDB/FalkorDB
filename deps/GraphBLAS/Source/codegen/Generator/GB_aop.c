//------------------------------------------------------------------------------
// GB_aop:  assign/subassign kernels with accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C(I,J)<M> += A

#include "GB_control.h"
GB_type_enabled
#if GB_TYPE_ENABLED
#include "GB.h"
#include "FactoryKernels/GB_aop__include.h"

// accum operator
GB_accumop
GB_ztype
GB_xtype
GB_ytype
GB_declarey
GB_copy_aij_to_ywork

// A and C matrices
GB_atype
GB_ctype
GB_declarec
GB_copy_aij_to_cwork
GB_copy_aij_to_c
GB_copy_cwork_to_c
GB_ax_mask

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
GB_disable

#include "assign/include/GB_assign_shared_definitions.h"

//------------------------------------------------------------------------------
// C += A, accumulate a sparse matrix into a dense matrix
//------------------------------------------------------------------------------

#undef  GB_SCALAR_ASSIGN
#define GB_SCALAR_ASSIGN 0

GrB_Info GB (_subassign_23)
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

GrB_Info GB (_subassign_22)
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

