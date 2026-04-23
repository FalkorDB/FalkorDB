//------------------------------------------------------------------------------
// GB_jit__emult_bitmap__000000ee60ee68b__LG_MF_CreateCompareVec32.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.4.0, Timothy A. Davis, (c) 2017-2026,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: LG_MF_CreateCompareVec32, ztype: LG_MF_compareTuple32, xtype: LG_MF_resultTuple32, ytype: int32_t

typedef struct{ double residual; int32_t di; int32_t dj; int32_t j; int32_t unused; } LG_MF_compareTuple32;
#define GB_LG_MF_compareTuple32_USER_DEFN \
"typedef struct{ double residual; int32_t di; int32_t dj; int32_t j; int32_t unused; } LG_MF_compareTuple32;"

typedef struct{ double residual; int32_t d; int32_t j; } LG_MF_resultTuple32;
#define GB_LG_MF_resultTuple32_USER_DEFN \
"typedef struct{ double residual; int32_t d; int32_t j; } LG_MF_resultTuple32;"

// binary operator types:
#define GB_Z_TYPE LG_MF_compareTuple32
#define GB_X_TYPE LG_MF_resultTuple32
#define GB_Y_TYPE int32_t

// binary operator:
#ifndef GB_GUARD_LG_MF_CreateCompareVec32_DEFINED
#define GB_GUARD_LG_MF_CreateCompareVec32_DEFINED
GB_STATIC_INLINE
void LG_MF_CreateCompareVec32(LG_MF_compareTuple32 *comp, const LG_MF_resultTuple32 *res, const int32_t *height) { comp->di = (*height); comp->residual = res->residual; comp->dj = res->d; comp->j = res->j; comp->unused = 0 ; }
#define GB_LG_MF_CreateCompareVec32_USER_DEFN \
"void LG_MF_CreateCompareVec32(LG_MF_compareTuple32 *comp, const LG_MF_resultTuple32 *res, const int32_t *height) { comp->di = (*height); comp->residual = res->residual; comp->dj = res->d; comp->j = res->j; comp->unused = 0 ; }"
#endif
#define GB_BINOP(z,x,y,i,j)  LG_MF_CreateCompareVec32 (&(z), &(x), &(y))
#define GB_COPY_A_to_C(Cx,pC,Ax,pA,A_iso)
#define GB_COPY_B_to_C(Cx,pC,Bx,pB,B_iso)

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
#define GB_C_TYPE LG_MF_compareTuple32
#define GB_PUTC(c,Cx,p) Cx [p] = c
#define GB_Cp_TYPE uint64_t
#define GB_Cj_TYPE uint64_t
#define GB_Cj_SIGNED_TYPE int64_t
#define GB_Ci_TYPE uint64_t
#define GB_Ci_SIGNED_TYPE int64_t
#define GB_Cp_BITS 64
#define GB_Cj_BITS 64
#define GB_Ci_BITS 64
#define GB_EWISEOP(Cx,p,aij,bij,i,j) GB_BINOP (Cx [p], aij, bij, i, j)

// M matrix: none
#define GB_M_TYPE void
#define GB_MCAST(Mx,p,msize) 1
#define GB_MASK_STRUCT 1
#define GB_MASK_COMP   0
#define GB_NO_MASK     1
#define GB_Mp_TYPE uint64_t
#define GB_Mj_TYPE uint64_t
#define GB_Mj_SIGNED_TYPE int64_t
#define GB_Mi_TYPE uint64_t
#define GB_Mi_SIGNED_TYPE int64_t
#define GB_Mp_BITS 64
#define GB_Mj_BITS 64
#define GB_Mi_BITS 64

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
#define GB_A_ISO 0
#define GB_A_TYPE LG_MF_resultTuple32
#define GB_A2TYPE LG_MF_resultTuple32
#define GB_DECLAREA(a) LG_MF_resultTuple32 a
#define GB_GETA(a,Ax,p,iso) a = Ax [p]
#define GB_Ap_TYPE uint64_t
#define GB_Aj_TYPE uint64_t
#define GB_Aj_SIGNED_TYPE int64_t
#define GB_Ai_TYPE uint64_t
#define GB_Ai_SIGNED_TYPE int64_t
#define GB_Ap_BITS 64
#define GB_Aj_BITS 64
#define GB_Ai_BITS 64

// B matrix: full
#define GB_B_IS_HYPER  0
#define GB_B_IS_SPARSE 0
#define GB_B_IS_BITMAP 0
#define GB_B_IS_FULL   1
#define GBp_B(Bp,k,vlen) ((k) * (vlen))
#define GBh_B(Bh,k)      (k)
#define GBi_B(Bi,p,vlen) ((p) % (vlen))
#define GBb_B(Bb,p)      1
#define GB_B_NVALS(e) int64_t e = (B->vlen * B->vdim)
#define GB_B_NHELD(e) GB_B_NVALS(e)
#define GB_B_ISO 0
#define GB_B_TYPE int32_t
#define GB_B2TYPE int32_t
#define GB_DECLAREB(b) int32_t b
#define GB_GETB(b,Bx,p,iso) b = Bx [p]
#define GB_Bp_TYPE uint64_t
#define GB_Bj_TYPE uint64_t
#define GB_Bj_SIGNED_TYPE int64_t
#define GB_Bi_TYPE uint64_t
#define GB_Bi_SIGNED_TYPE int64_t
#define GB_Bp_BITS 64
#define GB_Bj_BITS 64
#define GB_Bi_BITS 64

#include "include/GB_ewise_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__emult_bitmap__000000ee60ee68b__LG_MF_CreateCompareVec32
#define GB_jit_query  GB_jit__emult_bitmap__000000ee60ee68b__LG_MF_CreateCompareVec32_query
#endif
#include "template/GB_jit_kernel_emult_bitmap.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0xeddfc49ad0ce1061 ;
    v [0] = 10 ; v [1] = 4 ; v [2] = 0 ;
    defn [0] = GB_LG_MF_CreateCompareVec32_USER_DEFN ;
    defn [1] = NULL ;
    defn [2] = GB_LG_MF_compareTuple32_USER_DEFN ;
    defn [3] = GB_LG_MF_resultTuple32_USER_DEFN ;
    defn [4] = NULL ;
    return (true) ;
}
