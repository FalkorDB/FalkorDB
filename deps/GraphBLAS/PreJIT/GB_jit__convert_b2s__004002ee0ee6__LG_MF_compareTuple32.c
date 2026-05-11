//------------------------------------------------------------------------------
// GB_jit__convert_b2s__004002ee0ee6__LG_MF_compareTuple32.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.3.1, Timothy A. Davis, (c) 2017-2026,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: LG_MF_compareTuple32, ztype: LG_MF_compareTuple32, xtype: LG_MF_compareTuple32, ytype: void

typedef struct{ double residual; int32_t di; int32_t dj; int32_t j; int32_t unused; } LG_MF_compareTuple32;
#define GB_LG_MF_compareTuple32_USER_DEFN \
"typedef struct{ double residual; int32_t di; int32_t dj; int32_t j; int32_t unused; } LG_MF_compareTuple32;"

// unary operator types:
#define GB_Z_TYPE LG_MF_compareTuple32
#define GB_X_TYPE LG_MF_compareTuple32
#define GB_Y_TYPE void
#define GB_DECLAREZ(zwork) LG_MF_compareTuple32 zwork
#define GB_DECLAREX(xwork) LG_MF_compareTuple32 xwork
#define GB_DECLAREY(ywork) void ywork

// unary operator:
#define GB_UNARYOP(z,x,i,j,y) z = x
#define GB_DEPENDS_ON_X 1
#define GB_DEPENDS_ON_Y 0
#define GB_DEPENDS_ON_I 0
#define GB_DEPENDS_ON_J 0
#define GB_UNOP(Cx,pC,Ax,pA,A_iso,i,j,y) Cx [pC] = Ax [pA]

// C type:
#define GB_C_TYPE LG_MF_compareTuple32
#define GB_Cp_TYPE uint64_t
#define GB_Cj_TYPE uint64_t
#define GB_Cj_SIGNED_TYPE int64_t
#define GB_Ci_TYPE uint64_t
#define GB_Ci_SIGNED_TYPE int64_t
#define GB_Cp_BITS 64
#define GB_Cj_BITS 64
#define GB_Ci_BITS 64

// A matrix: bitmap
#define GB_A_IS_HYPER  0
#define GB_A_IS_SPARSE 0
#define GB_A_IS_BITMAP 1
#define GB_A_IS_FULL   0
#define GBp_A(Ap,k,vlen) ((k) * (vlen))
#define GBh_A(Ah,k)      (k)
#define GBi_A(Ai,p,vlen) ((p) % (vlen))
#define GBb_A(Ab,p)      Ab [p]
#define GB_A_NVALS(e) int64_t e = A->nvals
#define GB_A_NHELD(e) int64_t e = (A->vlen * A->vdim)
#define GB_A_HAS_ZOMBIES 0
#define GB_A_ISO 0
#define GB_A_TYPE LG_MF_compareTuple32
#define GB_A2TYPE LG_MF_compareTuple32
#define GB_DECLAREA(a) LG_MF_compareTuple32 a
#define GB_GETA(a,Ax,p,iso) a = Ax [p]
#define GB_Ap_TYPE uint64_t
#define GB_Aj_TYPE uint64_t
#define GB_Aj_SIGNED_TYPE int64_t
#define GB_Ai_TYPE uint64_t
#define GB_Ai_SIGNED_TYPE int64_t
#define GB_Ap_BITS 64
#define GB_Aj_BITS 64
#define GB_Ai_BITS 64

#include "include/GB_kernel_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__convert_b2s__004002ee0ee6__LG_MF_compareTuple32
#define GB_jit_query  GB_jit__convert_b2s__004002ee0ee6__LG_MF_compareTuple32_query
#endif
#include "template/GB_jit_kernel_convert_b2s.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x2b4cb6e124b6cc4c ;
    v [0] = 10 ; v [1] = 3 ; v [2] = 1 ;
    defn [0] = NULL ;
    defn [1] = NULL ;
    defn [2] = GB_LG_MF_compareTuple32_USER_DEFN ;
    defn [3] = defn [2] ;
    defn [4] = NULL ;
    return (true) ;
}
