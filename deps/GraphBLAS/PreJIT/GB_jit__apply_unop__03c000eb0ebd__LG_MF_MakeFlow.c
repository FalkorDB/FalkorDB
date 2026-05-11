//------------------------------------------------------------------------------
// GB_jit__apply_unop__03c000eb0ebd__LG_MF_MakeFlow.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.3.1, Timothy A. Davis, (c) 2017-2026,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: LG_MF_MakeFlow, ztype: LG_MF_flowEdge, xtype: double, ytype: void

typedef struct{ double capacity; double flow; } LG_MF_flowEdge;
#define GB_LG_MF_flowEdge_USER_DEFN \
"typedef struct{ double capacity; double flow; } LG_MF_flowEdge;"

// unary operator types:
#define GB_Z_TYPE LG_MF_flowEdge
#define GB_X_TYPE double
#define GB_Y_TYPE void
#define GB_DECLAREZ(zwork) LG_MF_flowEdge zwork
#define GB_DECLAREX(xwork) double xwork
#define GB_DECLAREY(ywork) void ywork

// unary operator:
#ifndef GB_GUARD_LG_MF_MakeFlow_DEFINED
#define GB_GUARD_LG_MF_MakeFlow_DEFINED
GB_STATIC_INLINE
void LG_MF_MakeFlow(LG_MF_flowEdge * flow_edge, const double * flow){ flow_edge->capacity = 0; flow_edge->flow = (*flow); }
#define GB_LG_MF_MakeFlow_USER_DEFN \
"void LG_MF_MakeFlow(LG_MF_flowEdge * flow_edge, const double * flow){ flow_edge->capacity = 0; flow_edge->flow = (*flow); }"
#endif
#define GB_UNARYOP(z,x,i,j,y)  LG_MF_MakeFlow (&(z), &(x))
#define GB_DEPENDS_ON_X 1
#define GB_DEPENDS_ON_Y 0
#define GB_DEPENDS_ON_I 0
#define GB_DEPENDS_ON_J 0
#define GB_UNOP(Cx,pC,Ax,pA,A_iso,i,j,y) GB_UNARYOP (Cx [pC], Ax [pA],  ,  ,  )

// C type:
#define GB_C_TYPE LG_MF_flowEdge
#define GB_Cp_TYPE uint64_t
#define GB_Cj_TYPE uint64_t
#define GB_Cj_SIGNED_TYPE int64_t
#define GB_Ci_TYPE uint64_t
#define GB_Ci_SIGNED_TYPE int64_t
#define GB_Cp_BITS 64
#define GB_Cj_BITS 64
#define GB_Ci_BITS 64

// A matrix: sparse
#define GB_A_IS_HYPER  0
#define GB_A_IS_SPARSE 1
#define GB_A_IS_BITMAP 0
#define GB_A_IS_FULL   0
#define GBp_A(Ap,k,vlen) Ap [k]
#define GBh_A(Ah,k)      (k)
#define GBi_A(Ai,p,vlen) Ai [p]
#define GBb_A(Ab,p)      1
#define GB_A_NVALS(e) int64_t e = A->nvals
#define GB_A_NHELD(e) GB_A_NVALS(e)
#define GB_A_HAS_ZOMBIES 0
#define GB_A_ISO 0
#define GB_A_TYPE double
#define GB_A2TYPE double
#define GB_DECLAREA(a) double a
#define GB_GETA(a,Ax,p,iso) a = Ax [p]
#define GB_Ap_TYPE uint32_t
#define GB_Aj_TYPE uint32_t
#define GB_Aj_SIGNED_TYPE int32_t
#define GB_Ai_TYPE uint32_t
#define GB_Ai_SIGNED_TYPE int32_t
#define GB_Ap_BITS 32
#define GB_Aj_BITS 32
#define GB_Ai_BITS 32

#include "include/GB_kernel_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__apply_unop__03c000eb0ebd__LG_MF_MakeFlow
#define GB_jit_query  GB_jit__apply_unop__03c000eb0ebd__LG_MF_MakeFlow_query
#endif
#include "template/GB_jit_kernel_apply_unop.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0xd861071877fba915 ;
    v [0] = 10 ; v [1] = 3 ; v [2] = 1 ;
    defn [0] = GB_LG_MF_MakeFlow_USER_DEFN ;
    defn [1] = NULL ;
    defn [2] = GB_LG_MF_flowEdge_USER_DEFN ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
