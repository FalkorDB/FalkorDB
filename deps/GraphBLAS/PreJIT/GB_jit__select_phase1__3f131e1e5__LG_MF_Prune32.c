//------------------------------------------------------------------------------
// GB_jit__select_phase1__3f131e1e5__LG_MF_Prune32.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.3.1, Timothy A. Davis, (c) 2017-2026,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: LG_MF_Prune32, ztype: bool, xtype: LG_MF_resultTuple32, ytype: bool

typedef struct{ double residual; int32_t d; int32_t j; } LG_MF_resultTuple32;
#define GB_LG_MF_resultTuple32_USER_DEFN \
"typedef struct{ double residual; int32_t d; int32_t j; } LG_MF_resultTuple32;"

// unary operator types:
#define GB_Z_TYPE bool
#define GB_X_TYPE LG_MF_resultTuple32
#define GB_Y_TYPE bool

// index unary operator:
#ifndef GB_GUARD_LG_MF_Prune32_DEFINED
#define GB_GUARD_LG_MF_Prune32_DEFINED
GB_STATIC_INLINE
void LG_MF_Prune32(bool * z, const LG_MF_resultTuple32 * x, GrB_Index ix, GrB_Index jx, const bool * theta){ *z = (x->j != -1) ; }
#define GB_LG_MF_Prune32_USER_DEFN \
"void LG_MF_Prune32(bool * z, const LG_MF_resultTuple32 * x, GrB_Index ix, GrB_Index jx, const bool * theta){ *z = (x->j != -1) ; }"
#endif
#define GB_IDXUNOP(z,x,i,j,y) LG_MF_Prune32 (&(z), &(x), i, j, &(y))
#define GB_DEPENDS_ON_X 1
#define GB_DEPENDS_ON_I 1
#define GB_DEPENDS_ON_J 1
#define GB_DEPENDS_ON_Y 1
#define GB_ENTRY_SELECTOR

// test if A(i,j) is to be kept:
#define GB_TEST_VALUE_OF_ENTRY(keep,p) \
    bool keep ;                        \
    GB_IDXUNOP (keep, Ax [p], i, j, y) ;

// copy A(i,j) to C(i,j):
#define GB_SELECT_ENTRY(Cx,pC,Ax,pA) Cx [pC] = Ax [pA]

// C matrix: sparse
#define GB_C_IS_HYPER  0
#define GB_C_IS_SPARSE 1
#define GB_C_IS_BITMAP 0
#define GB_C_IS_FULL   0
#define GBp_C(Cp,k,vlen) Cp [k]
#define GBh_C(Ch,k)      (k)
#define GBi_C(Ci,p,vlen) Ci [p]
#define GBb_C(Cb,p)      1
#define GB_C_NVALS(e) int64_t e = C->nvals
#define GB_C_NHELD(e) GB_C_NVALS(e)
#define GB_C_ISO 0
#define GB_C_IN_ISO 0
#define GB_C_TYPE LG_MF_resultTuple32
#define GB_PUTC(c,Cx,p) Cx [p] = c
#define GB_Cp_TYPE uint32_t
#define GB_Cj_TYPE uint32_t
#define GB_Cj_SIGNED_TYPE int32_t
#define GB_Ci_TYPE uint32_t
#define GB_Ci_SIGNED_TYPE int32_t
#define GB_Cp_BITS 32
#define GB_Cj_BITS 32
#define GB_Ci_BITS 32

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

#include "include/GB_select_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__select_phase1__3f131e1e5__LG_MF_Prune32
#define GB_jit_query  GB_jit__select_phase1__3f131e1e5__LG_MF_Prune32_query
#endif
#include "template/GB_jit_kernel_select_phase1.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0xabe5f98fe96f0dde ;
    v [0] = 10 ; v [1] = 3 ; v [2] = 1 ;
    defn [0] = GB_LG_MF_Prune32_USER_DEFN ;
    defn [1] = NULL ;
    defn [2] = GB_LG_MF_resultTuple32_USER_DEFN ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
