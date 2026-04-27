//------------------------------------------------------------------------------
// GB_jit__apply_unop__03c000be0bed__LG_MF_ResidualFlow32.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.3.1, Timothy A. Davis, (c) 2017-2026,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: LG_MF_ResidualFlow32, ztype: double, xtype: LG_MF_resultTuple32, ytype: void

typedef struct{ double residual; int32_t d; int32_t j; } LG_MF_resultTuple32;
#define GB_LG_MF_resultTuple32_USER_DEFN \
"typedef struct{ double residual; int32_t d; int32_t j; } LG_MF_resultTuple32;"

// unary operator types:
#define GB_Z_TYPE double
#define GB_X_TYPE LG_MF_resultTuple32
#define GB_Y_TYPE void
#define GB_DECLAREZ(zwork) double zwork
#define GB_DECLAREX(xwork) LG_MF_resultTuple32 xwork
#define GB_DECLAREY(ywork) void ywork

// unary operator:
#ifndef GB_GUARD_LG_MF_ResidualFlow32_DEFINED
#define GB_GUARD_LG_MF_ResidualFlow32_DEFINED
GB_STATIC_INLINE
void LG_MF_ResidualFlow32(double *z, const LG_MF_resultTuple32 *x) { (*z) = x->residual; }
#define GB_LG_MF_ResidualFlow32_USER_DEFN \
"void LG_MF_ResidualFlow32(double *z, const LG_MF_resultTuple32 *x) { (*z) = x->residual; }"
#endif
#define GB_UNARYOP(z,x,i,j,y)  LG_MF_ResidualFlow32 (&(z), &(x))
#define GB_DEPENDS_ON_X 1
#define GB_DEPENDS_ON_Y 0
#define GB_DEPENDS_ON_I 0
#define GB_DEPENDS_ON_J 0
#define GB_UNOP(Cx,pC,Ax,pA,A_iso,i,j,y) GB_UNARYOP (Cx [pC], Ax [pA],  ,  ,  )

// C type:
#define GB_C_TYPE double
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
#define GB_A_TYPE LG_MF_resultTuple32
#define GB_A2TYPE LG_MF_resultTuple32
#define GB_DECLAREA(a) LG_MF_resultTuple32 a
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
#define GB_jit_kernel GB_jit__apply_unop__03c000be0bed__LG_MF_ResidualFlow32
#define GB_jit_query  GB_jit__apply_unop__03c000be0bed__LG_MF_ResidualFlow32_query
#endif
#include "template/GB_jit_kernel_apply_unop.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x06a6d568d537539c ;
    v [0] = 10 ; v [1] = 3 ; v [2] = 1 ;
    defn [0] = GB_LG_MF_ResidualFlow32_USER_DEFN ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = GB_LG_MF_resultTuple32_USER_DEFN ;
    defn [4] = NULL ;
    return (true) ;
}
