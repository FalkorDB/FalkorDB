//------------------------------------------------------------------------------
// GB_jit__subassign_04__7f1c505111101100.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.3.0, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// subassign: C pair= A 
#define GB_ASSIGN_KIND GB_SUBASSIGN
#define GB_SCALAR_ASSIGN 0
#define GB_I_KIND GB_ALL
#define GB_J_KIND GB_ALL
#define GB_I_TYPE uint64_t
#define GB_J_TYPE uint64_t
#define GB_I_IS_32 0
#define GB_J_IS_32 0
#define GB_C_REPLACE 0

// accum: (pair, bool)

// accum operator types:
#define GB_Z_TYPE bool
#define GB_X_TYPE bool
#define GB_Y_TYPE bool
#define GB_DECLAREZ(zwork) bool zwork
#define GB_DECLAREX(xwork) bool xwork
#define GB_DECLAREY(ywork) bool ywork

// accum operator:
#define GB_ACCUM_OP(z,x,y) z = 1
#define GB_UPDATE(z,y) GB_ACCUM_OP(z,z,y)
#define GB_ACCUMULATE_aij(Cx,pC,Ax,pA,A_iso,ywork,C_iso) \
{                                          \
    GB_UPDATE (Cx [pC], ywork) ;          \
}
#define GB_ACCUMULATE_scalar(Cx,pC,ywork,C_iso) /* unused */

// C matrix: hypersparse
#define GB_C_IS_HYPER  1
#define GB_C_IS_SPARSE 0
#define GB_C_IS_BITMAP 0
#define GB_C_IS_FULL   0
#define GBp_C(Cp,k,vlen) Cp [k]
#define GBh_C(Ch,k)      Ch [k]
#define GBi_C(Ci,p,vlen) Ci [p]
#define GBb_C(Cb,p)      1
#define GB_C_NVALS(e) int64_t e = C->nvals
#define GB_C_NHELD(e) GB_C_NVALS(e)
#define GB_C_ISO 0
#define GB_C_IN_ISO 0
#define GB_C_TYPE bool
#define GB_PUTC(zwork,Cx,p) Cx [p] = zwork
#define GB_Cp_TYPE uint32_t
#define GB_Cj_TYPE uint32_t
#define GB_Cj_SIGNED_TYPE int32_t
#define GB_Ci_TYPE uint32_t
#define GB_Ci_SIGNED_TYPE int32_t
#define GB_Cp_BITS 32
#define GB_Cj_BITS 32
#define GB_Ci_BITS 32
#define GB_DECLAREC(cwork) bool cwork
#define GB_COPY_A_to_C(Cx,pC,Ax,pA,A_iso) Cx [pC] = Ax [0]
#define GB_COPY_aij_to_C(Cx,pC,Ax,pA,A_iso,cwork,C_iso) Cx [pC] = cwork
#define GB_COPY_aij_to_cwork(cwork,Ax,p,A_iso) cwork = Ax [0]
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

// A matrix: hypersparse
#define GB_A_IS_HYPER  1
#define GB_A_IS_SPARSE 0
#define GB_A_IS_BITMAP 0
#define GB_A_IS_FULL   0
#define GBp_A(Ap,k,vlen) Ap [k]
#define GBh_A(Ah,k)      Ah [k]
#define GBi_A(Ai,p,vlen) Ai [p]
#define GBb_A(Ab,p)      1
#define GB_A_NVALS(e) int64_t e = A->nvals
#define GB_A_NHELD(e) GB_A_NVALS(e)
#define GB_A_ISO 1
#define GB_A_TYPE bool
#define GB_A2TYPE bool
#define GB_DECLAREA(a) bool a
#define GB_GETA(a,Ax,p,iso) a = Ax [0]
#define GB_Ap_TYPE uint32_t
#define GB_Aj_TYPE uint32_t
#define GB_Aj_SIGNED_TYPE int32_t
#define GB_Ai_TYPE uint32_t
#define GB_Ai_SIGNED_TYPE int32_t
#define GB_Ap_BITS 32
#define GB_Aj_BITS 32
#define GB_Ai_BITS 32
#define GB_COPY_aij_to_ywork(ywork,Ax,pA,A_iso) GB_GETA (ywork, Ax, pA, A_iso)
#define GB_COPY_scalar_to_ywork(ywork,scalar) /* unused */

// S matrix: hypersparse
#define GB_S_IS_HYPER  1
#define GB_S_IS_SPARSE 0
#define GB_S_IS_BITMAP 0
#define GB_S_IS_FULL   0
#define GBp_S(Sp,k,vlen) Sp [k]
#define GBh_S(Sh,k)      Sh [k]
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
#define GB_jit_kernel GB_jit__subassign_04__7f1c505111101100
#define GB_jit_query  GB_jit__subassign_04__7f1c505111101100_query
#endif
#include "template/GB_jit_kernel_subassign_04.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x35e611d1e67dc156 ;
    v [0] = 10 ; v [1] = 3 ; v [2] = 0 ;
    defn [0] = NULL ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
