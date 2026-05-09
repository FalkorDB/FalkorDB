//------------------------------------------------------------------------------
// GB_jit__emult_bitmap__000000eeb2eebaa__LG_MF_UpdateFlow.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.4.0, Timothy A. Davis, (c) 2017-2026,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: LG_MF_UpdateFlow, ztype: LG_MF_flowEdge, xtype: LG_MF_flowEdge, ytype: double

typedef struct{ double capacity; double flow; } LG_MF_flowEdge;
#define GB_LG_MF_flowEdge_USER_DEFN \
"typedef struct{ double capacity; double flow; } LG_MF_flowEdge;"

// binary operator types:
#define GB_Z_TYPE LG_MF_flowEdge
#define GB_X_TYPE LG_MF_flowEdge
#define GB_Y_TYPE double

// binary operator:
#ifndef GB_GUARD_LG_MF_UpdateFlow_DEFINED
#define GB_GUARD_LG_MF_UpdateFlow_DEFINED
GB_STATIC_INLINE
void LG_MF_UpdateFlow(LG_MF_flowEdge *z, const LG_MF_flowEdge *x, const double *y) { z->capacity = x->capacity; z->flow = x->flow + (*y); }
#define GB_LG_MF_UpdateFlow_USER_DEFN \
"void LG_MF_UpdateFlow(LG_MF_flowEdge *z, const LG_MF_flowEdge *x, const double *y) { z->capacity = x->capacity; z->flow = x->flow + (*y); }"
#endif
#define GB_BINOP(z,x,y,i,j)  LG_MF_UpdateFlow (&(z), &(x), &(y))
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
#define GB_C_TYPE LG_MF_flowEdge
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

// M matrix: bitmap
#define GB_M_IS_HYPER  0
#define GB_M_IS_SPARSE 0
#define GB_M_IS_BITMAP 1
#define GB_M_IS_FULL   0
#define GBp_M(Mp,k,vlen) ((k) * (vlen))
#define GBh_M(Mh,k)      (k)
#define GBi_M(Mi,p,vlen) ((p) % (vlen))
#define GBb_M(Mb,p)      Mb [p]
// structural mask:
#define GB_M_TYPE void
#define GB_MCAST(Mx,p,msize) 1
#define GB_MASK_STRUCT 1
#define GB_MASK_COMP   0
#define GB_NO_MASK     0
#define GB_M_NVALS(e) int64_t e = M->nvals
#define GB_M_NHELD(e) int64_t e = (M->vlen * M->vdim)
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
#define GB_A_TYPE LG_MF_flowEdge
#define GB_A2TYPE LG_MF_flowEdge
#define GB_DECLAREA(a) LG_MF_flowEdge a
#define GB_GETA(a,Ax,p,iso) a = Ax [p]
#define GB_Ap_TYPE uint64_t
#define GB_Aj_TYPE uint64_t
#define GB_Aj_SIGNED_TYPE int64_t
#define GB_Ai_TYPE uint64_t
#define GB_Ai_SIGNED_TYPE int64_t
#define GB_Ap_BITS 64
#define GB_Aj_BITS 64
#define GB_Ai_BITS 64

// B matrix: bitmap
#define GB_B_IS_HYPER  0
#define GB_B_IS_SPARSE 0
#define GB_B_IS_BITMAP 1
#define GB_B_IS_FULL   0
#define GBp_B(Bp,k,vlen) ((k) * (vlen))
#define GBh_B(Bh,k)      (k)
#define GBi_B(Bi,p,vlen) ((p) % (vlen))
#define GBb_B(Bb,p)      Bb [p]
#define GB_B_NVALS(e) int64_t e = B->nvals
#define GB_B_NHELD(e) int64_t e = (B->vlen * B->vdim)
#define GB_B_ISO 0
#define GB_B_TYPE double
#define GB_B2TYPE double
#define GB_DECLAREB(b) double b
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
#define GB_jit_kernel GB_jit__emult_bitmap__000000eeb2eebaa__LG_MF_UpdateFlow
#define GB_jit_query  GB_jit__emult_bitmap__000000eeb2eebaa__LG_MF_UpdateFlow_query
#endif
#include "template/GB_jit_kernel_emult_bitmap.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0xc8a807d090d41dd3 ;
    v [0] = 10 ; v [1] = 4 ; v [2] = 0 ;
    defn [0] = GB_LG_MF_UpdateFlow_USER_DEFN ;
    defn [1] = NULL ;
    defn [2] = GB_LG_MF_flowEdge_USER_DEFN ;
    defn [3] = defn [2] ;
    defn [4] = NULL ;
    return (true) ;
}
