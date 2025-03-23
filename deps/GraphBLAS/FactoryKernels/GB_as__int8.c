//------------------------------------------------------------------------------
// GB_as:  assign/subassign kernels with no accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C(I,J)<M> = A

#include "GB_control.h"
#if defined (GxB_NO_INT8)
#define GB_TYPE_ENABLED 0
#else
#define GB_TYPE_ENABLED 1
#endif

#if GB_TYPE_ENABLED
#include "GB.h"
#include "FactoryKernels/GB_as__include.h"

// A and C matrices
#define GB_A_TYPE int8_t
#define GB_C_TYPE int8_t
#define GB_DECLAREC(cwork) int8_t cwork
#define GB_COPY_aij_to_cwork(cwork,Ax,pA,A_iso) cwork = Ax [A_iso ? 0 : (pA)]
#define GB_COPY_aij_to_C(Cx,pC,Ax,pA,A_iso,cwork,C_iso) Cx [pC] = (A_iso) ? cwork : Ax [pA]
#define GB_COPY_cwork_to_C(Cx,pC,cwork,C_iso) Cx [pC] = cwork
#define GB_AX_MASK(Ax,pA,asize) (Ax [pA] != 0)

// disable this operator and use the generic case if these conditions hold
#if (defined(GxB_NO_INT8))
#define GB_DISABLE 1
#else
#define GB_DISABLE 0
#endif

#include "assign/include/GB_assign_shared_definitions.h"

//------------------------------------------------------------------------------
// C<M> = scalar, when C is dense
//------------------------------------------------------------------------------

#undef  GB_SCALAR_ASSIGN
#define GB_SCALAR_ASSIGN 1

GrB_Info GB (_subassign_05d__int8)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const GB_void *scalar,      // of type C->type, already typecasted
    GB_Werk Werk
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    GB_C_TYPE cwork = (*((GB_C_TYPE *) scalar)) ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    #include "assign/template/GB_subassign_05d_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C<A> = A, when C is dense
//------------------------------------------------------------------------------

#undef  GB_SCALAR_ASSIGN
#define GB_SCALAR_ASSIGN 0

GrB_Info GB (_subassign_06d__int8)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const bool Mask_struct,
    GB_Werk Werk
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    ASSERT (C->type == A->type) ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    #include "assign/template/GB_subassign_06d_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C<M> = A, when C is empty and A is dense
//------------------------------------------------------------------------------

GrB_Info GB (_subassign_25__int8)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const GrB_Matrix A,
    GB_Werk Werk
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    ASSERT (C->type == A->type) ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    #include "assign/template/GB_subassign_25_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

#else
GB_EMPTY_PLACEHOLDER
#endif

