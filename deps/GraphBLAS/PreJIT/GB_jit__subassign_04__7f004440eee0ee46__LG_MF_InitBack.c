//------------------------------------------------------------------------------
// GB_jit__subassign_04__7f004440eee0ee46__LG_MF_InitBack.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.4.0, Timothy A. Davis, (c) 2017-2026,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// subassign: C(I,:) op= A 
#define GB_ASSIGN_KIND GB_SUBASSIGN
#define GB_SCALAR_ASSIGN 0
#define GB_I_KIND GB_RANGE
#define GB_J_KIND GB_ALL
#define GB_I_TYPE uint64_t
#define GB_J_TYPE uint64_t
#define GB_I_IS_32 0
#define GB_J_IS_32 0
#define GB_C_REPLACE 0

// accum: LG_MF_InitBack, ztype: LG_MF_flowEdge, xtype: LG_MF_flowEdge, ytype: LG_MF_flowEdge

typedef struct{ double capacity; double flow; } LG_MF_flowEdge;
#define GB_LG_MF_flowEdge_USER_DEFN \
"typedef struct{ double capacity; double flow; } LG_MF_flowEdge;"

// accum operator types:
#define GB_Z_TYPE LG_MF_flowEdge
#define GB_X_TYPE LG_MF_flowEdge
#define GB_Y_TYPE LG_MF_flowEdge
#define GB_DECLAREZ(zwork) LG_MF_flowEdge zwork
#define GB_DECLAREX(xwork) LG_MF_flowEdge xwork
#define GB_DECLAREY(ywork) LG_MF_flowEdge ywork

// accum operator:
#ifndef GB_GUARD_LG_MF_InitBack_DEFINED
#define GB_GUARD_LG_MF_InitBack_DEFINED
GB_STATIC_INLINE
void LG_MF_InitBack(LG_MF_flowEdge * z, const LG_MF_flowEdge * x, const LG_MF_flowEdge * y){ z->capacity = x->capacity; z->flow = x->flow - y->flow; }
#define GB_LG_MF_InitBack_USER_DEFN \
"void LG_MF_InitBack(LG_MF_flowEdge * z, const LG_MF_flowEdge * x, const LG_MF_flowEdge * y){ z->capacity = x->capacity; z->flow = x->flow - y->flow; }"
#endif
#define GB_ACCUM_OP(z,x,y)  LG_MF_InitBack (&(z), &(x), &(y))
#define GB_UPDATE(z,y) GB_ACCUM_OP(z,z,y)
#define GB_ACCUMULATE_aij(Cx,pC,Ax,pA,A_iso,ywork,C_iso) \
{                                          \
    GB_UPDATE (Cx [pC], Ax [pA]) ;          \
}
#define GB_ACCUMULATE_scalar(Cx,pC,ywork,C_iso) /* unused */

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
#define GB_C_TYPE LG_MF_flowEdge
#define GB_PUTC(zwork,Cx,p) Cx [p] = zwork
#define GB_Cp_TYPE uint32_t
#define GB_Cj_TYPE uint32_t
#define GB_Cj_SIGNED_TYPE int32_t
#define GB_Ci_TYPE uint32_t
#define GB_Ci_SIGNED_TYPE int32_t
#define GB_Cp_BITS 32
#define GB_Cj_BITS 32
#define GB_Ci_BITS 32
#define GB_DECLAREC(cwork) LG_MF_flowEdge cwork
#define GB_COPY_A_to_C(Cx,pC,Ax,pA,A_iso) Cx [pC] = Ax [pA]
#define GB_COPY_aij_to_C(Cx,pC,Ax,pA,A_iso,cwork,C_iso) \
    GB_COPY_A_to_C (Cx, pC, Ax, pA, A_iso)
#define GB_COPY_aij_to_cwork(cwork,Ax,p,A_iso) cwork = Ax [p]
#define GB_COPY_cwork_to_C(Cx,pC,cwork,C_iso) /* unused */
#define GB_COPY_scalar_to_cwork(cwork,scalar) /* unused */

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
#define GB_COPY_aij_to_ywork(ywork,Ax,pA,A_iso) GB_GETA (ywork, Ax, pA, A_iso)
#define GB_COPY_scalar_to_ywork(ywork,scalar) /* unused */

// S matrix: sparse
#define GB_S_IS_HYPER  0
#define GB_S_IS_SPARSE 1
#define GB_S_IS_BITMAP 0
#define GB_S_IS_FULL   0
#define GBp_S(Sp,k,vlen) Sp [k]
#define GBh_S(Sh,k)      (k)
#define GBi_S(Si,p,vlen) Si [p]
#define GBb_S(Sb,p)      1
#define GB_S_CONSTRUCTED 1
#define GB_Sp_TYPE uint32_t
#define GB_Sj_TYPE uint32_t
#define GB_Sj_SIGNED_TYPE int32_t
#define GB_Si_TYPE uint32_t
#define GB_Si_SIGNED_TYPE int32_t
#define GB_Sp_BITS 32
#define GB_Sj_BITS 32
#define GB_Si_BITS 32
#define GB_Sx_BITS 32
#define GB_Sx_TYPE uint32_t

#include "include/GB_assign_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__subassign_04__7f004440eee0ee46__LG_MF_InitBack
#define GB_jit_query  GB_jit__subassign_04__7f004440eee0ee46__LG_MF_InitBack_query
#endif
#include "template/GB_jit_kernel_subassign_04.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x9961e8da7bc8180a ;
    v [0] = 10 ; v [1] = 4 ; v [2] = 0 ;
    defn [0] = GB_LG_MF_InitBack_USER_DEFN ;
    defn [1] = NULL ;
    defn [2] = GB_LG_MF_flowEdge_USER_DEFN ;
    defn [3] = defn [2] ;
    defn [4] = NULL ;
    return (true) ;
}
