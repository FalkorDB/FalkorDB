//------------------------------------------------------------------------------
// GB_jit__bitmap_assign_2_whole__0000a03f000301a0.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.3.0, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// assign: C<!M,bitmap,struct,replace> = scalar 
#define GB_ASSIGN_KIND GB_ASSIGN
#define GB_SCALAR_ASSIGN 1
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
#define GB_C_ISO 1
#define GB_C_IN_ISO 1
#define GB_C_TYPE void
#define GB_PUTC(cwork,Cx,p)
#define GB_Cp_TYPE uint64_t
#define GB_Cj_TYPE uint64_t
#define GB_Cj_SIGNED_TYPE int64_t
#define GB_Ci_TYPE uint64_t
#define GB_Ci_SIGNED_TYPE int64_t
#define GB_Cp_BITS 64
#define GB_Cj_BITS 64
#define GB_Ci_BITS 64
#define GB_DECLAREC(cwork) bool cwork
#define GB_COPY_scalar_to_cwork(cwork,scalar) cwork = (*((GB_A_TYPE *) scalar))
#define GB_COPY_cwork_to_C(Cx,pC,cwork,C_iso)
#define GB_COPY_aij_to_cwork(cwork,Ax,pA,A_iso) /* unused */
#define GB_COPY_aij_to_C(Cx,pC,Ax,pA,A_iso,cwork,C_iso) /* unused */

// M matrix: bitmap
#define GB_M_IS_HYPER  0
#define GB_M_IS_SPARSE 0
#define GB_M_IS_BITMAP 1
#define GB_M_IS_FULL   0
#define GBp_M(Mp,k,vlen) ((k) * (vlen))
#define GBh_M(Mh,k)      (k)
#define GBi_M(Mi,p,vlen) ((p) % (vlen))
#define GBb_M(Mb,p)      Mb [p]
// structural mask (complemented):
#define GB_M_TYPE void
#define GB_MCAST(Mx,p,msize) 1
#define GB_MASK_STRUCT 1
#define GB_MASK_COMP   1
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

// scalar:
#define GB_A_TYPE bool

// A matrix: unused
#define GB_A_IS_HYPER  0
#define GB_A_IS_SPARSE 0
#define GB_A_IS_BITMAP 0
#define GB_A_IS_FULL   1
#define GBp_A(Ap,k,vlen) 0
#define GBh_A(Ah,k)      (k)
#define GBi_A(Ai,p,vlen) 0
#define GBb_A(Ab,p)      1
#define GB_A_NVALS(e) int64_t e = 1 ; /* unused */
#define GB_A_NHELD(e) int64_t e = 1 ; /* unused */
#define GB_Ap_TYPE uint64_t
#define GB_Aj_TYPE uint64_t
#define GB_Aj_SIGNED_TYPE int64_t
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
#define GB_jit_kernel GB_jit__bitmap_assign_2_whole__0000a03f000301a0
#define GB_jit_query  GB_jit__bitmap_assign_2_whole__0000a03f000301a0_query
#endif
#include "template/GB_jit_kernel_bitmap_assign_2_whole.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x2ff4a9684a816a34 ;
    v [0] = 10 ; v [1] = 3 ; v [2] = 0 ;
    defn [0] = NULL ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
