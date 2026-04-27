//------------------------------------------------------------------------------
// GB_jit__subassign_23__00000040eee0eec3__LG_MSF_tupleMin_double.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.3.1, Timothy A. Davis, (c) 2017-2026,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// subassign: C op= A 
#define GB_ASSIGN_KIND GB_SUBASSIGN
#define GB_SCALAR_ASSIGN 0
#define GB_I_KIND GB_ALL
#define GB_J_KIND GB_ALL
#define GB_I_TYPE uint64_t
#define GB_J_TYPE uint64_t
#define GB_I_IS_32 0
#define GB_J_IS_32 0
#define GB_C_REPLACE 0

// accum: LG_MSF_tupleMin_double, ztype: LG_MSF_tuple_double, xtype: LG_MSF_tuple_double, ytype: LG_MSF_tuple_double

typedef struct { double wInt; uint64_t idx; } LG_MSF_tuple_double;
#define GB_LG_MSF_tuple_double_USER_DEFN \
"typedef struct { double wInt; uint64_t idx; } LG_MSF_tuple_double;"

// accum operator types:
#define GB_Z_TYPE LG_MSF_tuple_double
#define GB_X_TYPE LG_MSF_tuple_double
#define GB_Y_TYPE LG_MSF_tuple_double
#define GB_DECLAREZ(zwork) LG_MSF_tuple_double zwork
#define GB_DECLAREX(xwork) LG_MSF_tuple_double xwork
#define GB_DECLAREY(ywork) LG_MSF_tuple_double ywork

// accum operator:
#ifndef GB_GUARD_LG_MSF_tupleMin_double_DEFINED
#define GB_GUARD_LG_MSF_tupleMin_double_DEFINED
GB_STATIC_INLINE
void LG_MSF_tupleMin_double ( LG_MSF_tuple_double *z, const LG_MSF_tuple_double *x, const LG_MSF_tuple_double *y ) { _Bool xSmaller = x->wInt < y->wInt || (x->wInt == y->wInt && x->idx < y->idx); z->wInt = (xSmaller)? x->wInt: y->wInt; z->idx = (xSmaller)? x->idx: y->idx; }
#define GB_LG_MSF_tupleMin_double_USER_DEFN \
"void LG_MSF_tupleMin_double ( LG_MSF_tuple_double *z, const LG_MSF_tuple_double *x, const LG_MSF_tuple_double *y ) { _Bool xSmaller = x->wInt < y->wInt || (x->wInt == y->wInt && x->idx < y->idx); z->wInt = (xSmaller)? x->wInt: y->wInt; z->idx = (xSmaller)? x->idx: y->idx; }"
#endif
#define GB_ACCUM_OP(z,x,y)  LG_MSF_tupleMin_double (&(z), &(x), &(y))
#define GB_UPDATE(z,y) GB_ACCUM_OP(z,z,y)
#define GB_ACCUMULATE_aij(Cx,pC,Ax,pA,A_iso,ywork,C_iso) \
{                                          \
    GB_UPDATE (Cx [pC], Ax [pA]) ;          \
}
#define GB_ACCUMULATE_scalar(Cx,pC,ywork,C_iso) /* unused */

// C matrix: full
#define GB_C_IS_HYPER  0
#define GB_C_IS_SPARSE 0
#define GB_C_IS_BITMAP 0
#define GB_C_IS_FULL   1
#define GBp_C(Cp,k,vlen) ((k) * (vlen))
#define GBh_C(Ch,k)      (k)
#define GBi_C(Ci,p,vlen) ((p) % (vlen))
#define GBb_C(Cb,p)      1
#define GB_C_NVALS(e) int64_t e = (C->vlen * C->vdim)
#define GB_C_NHELD(e) GB_C_NVALS(e)
#define GB_C_ISO 0
#define GB_C_IN_ISO 0
#define GB_C_TYPE LG_MSF_tuple_double
#define GB_PUTC(zwork,Cx,p) Cx [p] = zwork
#define GB_Cp_TYPE uint64_t
#define GB_Cj_TYPE uint64_t
#define GB_Cj_SIGNED_TYPE int64_t
#define GB_Ci_TYPE uint64_t
#define GB_Ci_SIGNED_TYPE int64_t
#define GB_Cp_BITS 64
#define GB_Cj_BITS 64
#define GB_Ci_BITS 64
#define GB_DECLAREC(cwork) LG_MSF_tuple_double cwork
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
#define GB_A_TYPE LG_MSF_tuple_double
#define GB_A2TYPE LG_MSF_tuple_double
#define GB_DECLAREA(a) LG_MSF_tuple_double a
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

// S matrix: not constructed
#define GB_S_CONSTRUCTED 0

#include "include/GB_assign_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__subassign_23__00000040eee0eec3__LG_MSF_tupleMin_double
#define GB_jit_query  GB_jit__subassign_23__00000040eee0eec3__LG_MSF_tupleMin_double_query
#endif
#include "template/GB_jit_kernel_subassign_23.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x6f95f3d092fbd503 ;
    v [0] = 10 ; v [1] = 3 ; v [2] = 1 ;
    defn [0] = GB_LG_MSF_tupleMin_double_USER_DEFN ;
    defn [1] = NULL ;
    defn [2] = GB_LG_MSF_tuple_double_USER_DEFN ;
    defn [3] = defn [2] ;
    defn [4] = NULL ;
    return (true) ;
}
