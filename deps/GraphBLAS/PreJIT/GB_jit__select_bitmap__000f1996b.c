//------------------------------------------------------------------------------
// GB_jit__select_bitmap__000f1996b.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.5.2, Timothy A. Davis, (c) 2017-2026,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: (valuegt, uint64_t)

// unary operator types:
#define GB_Z_TYPE bool
#define GB_X_TYPE uint64_t
#define GB_Y_TYPE uint64_t

// index unary operator:
#define GB_IDXUNOP(z,x,i,j,y) z = ((x) > (y))
#define GB_DEPENDS_ON_X 1
#define GB_DEPENDS_ON_I 0
#define GB_DEPENDS_ON_J 0
#define GB_DEPENDS_ON_Y 1
#define GB_ENTRY_SELECTOR

// test if A(i,j) is to be kept:
#define GB_TEST_VALUE_OF_ENTRY(keep,p) \
    bool keep ;                        \
    GB_DECLAREA (x) ;                  \
    GB_GETA (x, Ax, p, ) ;             \
    GB_IDXUNOP (keep, x, i, j, y) ;

// copy A(i,j) to C(i,j):
#define GB_SELECT_ENTRY(Cx,pC,Ax,pA) Cx [pC] = Ax [pA]

// C matrix: bitmap
#define GB_C_IS_HYPER  0
#define GB_C_IS_SPARSE 0
#define GB_C_IS_BITMAP 1
#define GB_C_IS_FULL   0
#define GBp_C(Cp,k,vlen) ((k) * (vlen))
#define GBh_C(Ch,k)      (k)
#define GBi_C(Ci,p,vlen) ((p) % (vlen))
#define GBb_C(Cb,p)      Cb [p]
#define GB_C_NVALS(e) int64_t e = C->nvals
#define GB_C_NHELD(e) int64_t e = (C->vlen * C->vdim)
#define GB_C_ISO 0
#define GB_C_IN_ISO 0
#define GB_C_TYPE int32_t
#define GB_PUTC(c,Cx,p) Cx [p] = c
#define GB_Cp_TYPE uint64_t
#define GB_Cj_TYPE uint64_t
#define GB_Cj_SIGNED_TYPE int64_t
#define GB_Ci_TYPE uint64_t
#define GB_Ci_SIGNED_TYPE int64_t
#define GB_Cp_BITS 64
#define GB_Cj_BITS 64
#define GB_Ci_BITS 64

// A matrix: full
#define GB_A_IS_HYPER  0
#define GB_A_IS_SPARSE 0
#define GB_A_IS_BITMAP 0
#define GB_A_IS_FULL   1
#define GBp_A(Ap,k,vlen) ((k) * (vlen))
#define GBh_A(Ah,k)      (k)
#define GBi_A(Ai,p,vlen) ((p) % (vlen))
#define GBb_A(Ab,p)      1
#define GB_A_NVALS(e) int64_t e = (A->vlen * A->vdim)
#define GB_A_NHELD(e) GB_A_NVALS(e)
#define GB_A_ISO 0
#define GB_A_TYPE int32_t
#define GB_A2TYPE uint64_t
#define GB_DECLAREA(a) uint64_t a
#define GB_GETA(a,Ax,p,iso) a = (uint64_t) (Ax [p])
#define GB_Ap_TYPE uint64_t
#define GB_Aj_TYPE uint64_t
#define GB_Aj_SIGNED_TYPE int64_t
#define GB_Ai_TYPE uint64_t
#define GB_Ai_SIGNED_TYPE int64_t
#define GB_Ap_BITS 64
#define GB_Aj_BITS 64
#define GB_Ai_BITS 64

#include "include/GB_select_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__select_bitmap__000f1996b
#define GB_jit_query  GB_jit__select_bitmap__000f1996b_query
#endif
#include "template/GB_jit_kernel_select_bitmap.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0xcf8094a82ff70d5d ;
    v [0] = 10 ; v [1] = 5 ; v [2] = 2 ;
    defn [0] = NULL ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
