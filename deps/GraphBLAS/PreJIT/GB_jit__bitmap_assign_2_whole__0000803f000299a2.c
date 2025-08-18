//------------------------------------------------------------------------------
// GB_jit__bitmap_assign_2_whole__0000803f000299a2.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.0.2, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// assign: C<M,bitmap,struct,replace> = A 
#define GB_ASSIGN_KIND GB_ASSIGN
#define GB_SCALAR_ASSIGN 0
#define GB_I_KIND GB_ALL
#define GB_J_KIND GB_ALL
#define GB_I_TYPE uint64_t
#define GB_J_TYPE uint64_t
#define GB_I_IS_32 0
#define GB_J_IS_32 0
#define GB_C_REPLACE 1

// accum: not present

#define GB_ACCUMULATE_scalar(Cx,pC,ywork,C_iso) /* unused */
#define GB_ACCUMULATE_aij(Cx,pC,Ax,pA,A_iso,ywork,C_iso) /* unused */

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
#define GB_C_TYPE uint64_t
#define GB_PUTC(cwork,Cx,p) Cx [p] = cwork
#define GB_Cp_TYPE uint64_t
#define GB_Cj_TYPE uint64_t
#define GB_Ci_TYPE uint64_t
#define GB_Ci_SIGNED_TYPE int64_t
#define GB_Cp_BITS 64
#define GB_Cj_BITS 64
#define GB_Ci_BITS 64
#define GB_DECLAREC(cwork) uint64_t cwork
#define GB_COPY_A_to_C(Cx,pC,Ax,pA,A_iso) Cx [pC] = Ax [pA]
#define GB_COPY_aij_to_C(Cx,pC,Ax,pA,A_iso,cwork,C_iso) \
    GB_COPY_A_to_C (Cx, pC, Ax, pA, A_iso)
#define GB_COPY_aij_to_cwork(cwork,Ax,p,A_iso) cwork = Ax [p]
#define GB_COPY_cwork_to_C(Cx,pC,cwork,C_iso) /* unused */
#define GB_COPY_scalar_to_cwork(cwork,scalar) /* unused */

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
#define GB_A_TYPE uint64_t
#define GB_A2TYPE void
#define GB_DECLAREA(a)
#define GB_GETA(a,Ax,p,iso)
#define GB_Ap_TYPE uint64_t
#define GB_Aj_TYPE uint64_t
#define GB_Ai_TYPE uint64_t
#define GB_Ai_SIGNED_TYPE int64_t
#define GB_Ap_BITS 64
#define GB_Aj_BITS 64
#define GB_Ai_BITS 64
#define GB_COPY_scalar_to_ywork(ywork,scalar) /* unused */
#define GB_COPY_aij_to_ywork(ywork,Ax,pA,A_iso) /* unused */

// S matrix: not constructed
#define GB_S_CONSTRUCTED 0

#include "include/GB_assign_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__bitmap_assign_2_whole__0000803f000299a2
#define GB_jit_query  GB_jit__bitmap_assign_2_whole__0000803f000299a2_query
#endif
#include "template/GB_jit_kernel_bitmap_assign_2_whole.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0xaa08797eb118a79b ;
    v [0] = 10 ; v [1] = 0 ; v [2] = 2 ;
    defn [0] = NULL ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
