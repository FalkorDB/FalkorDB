//------------------------------------------------------------------------------
// GB_aop:  assign/subassign kernels with accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C(I,J)<M> += A

#include "GB.h"
#include "GB_control.h"
#include "GB_ek_slice.h"
#include "GB_aop__include.h"

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
GB_copy_scalar_to_c
GB_ax_mask

// C(i,j) += ywork
#define GB_ACCUMULATE_scalar(Cx,pC,ywork) \
    GB_ACCUM_OP (Cx [pC], Cx [pC], ywork)

// C(i,j) += (ytype) A(i,j)
#define GB_ACCUMULATE_aij(Cx,pC,Ax,pA,A_iso,ywork)      \
{                                                       \
    if (A_iso)                                          \
    {                                                   \
        GB_ACCUMULATE_scalar (Cx, pC, ywork) ;          \
    }                                                   \
    else                                                \
    {                                                   \
        /* A and Y have the same type here */           \
        GB_ACCUMULATE_scalar (Cx, pC, Ax [pA]) ;        \
    }                                                   \
}

// disable this operator and use the generic case if these conditions hold
GB_disable

#include "GB_assign_shared_definitions.h"

//------------------------------------------------------------------------------
// C += A, accumulate a sparse matrix into a dense matrix
//------------------------------------------------------------------------------

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
    m4_divert(if_C_dense_update)
    { 
        int nthreads_max = GB_Context_nthreads_max ( ) ;
        double chunk = GB_Context_chunk ( ) ;
        #include "GB_subassign_23_template.c"
    }
    m4_divert(0)
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C += y, accumulate a scalar into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_subassign_22)
(
    GrB_Matrix C,
    const GB_void *ywork_handle
)
{
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    m4_divert(if_C_dense_update)
    { 
        // get the scalar ywork for C += ywork, of type GB_Y_TYPE
        GB_Y_TYPE ywork = (*((GB_Y_TYPE *) ywork_handle)) ;
        int nthreads_max = GB_Context_nthreads_max ( ) ;
        double chunk = GB_Context_chunk ( ) ;
        #include "GB_subassign_22_template.c"
        return (GrB_SUCCESS) ;
    }
    m4_divert(0)
    return (GrB_SUCCESS) ;
    #endif
}

