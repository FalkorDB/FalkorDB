//------------------------------------------------------------------------------
// GB_jit__subref_bitmap__0808c6f.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.3.0, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// subref: C=A(I,J) where C and A are bitmap/full
#define GB_I_KIND GB_LIST
#define GB_I_TYPE uint32_t
#define GB_J_KIND GB_ALL
#define GB_J_TYPE uint64_t

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
#define GB_C_TYPE int32_t
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
#define GB_A_TYPE int32_t
#define GB_Ap_TYPE uint64_t
#define GB_Aj_TYPE uint64_t
#define GB_Aj_SIGNED_TYPE int64_t
#define GB_Ai_TYPE uint64_t
#define GB_Ai_SIGNED_TYPE int64_t
#define GB_Ap_BITS 64
#define GB_Aj_BITS 64
#define GB_Ai_BITS 64

// R matrix: sparse
#define GB_R_IS_HYPER  0
#define GB_R_IS_SPARSE 1
#define GB_R_IS_BITMAP 0
#define GB_R_IS_FULL   0
#define GBp_R(Rp,k,vlen) Rp [k]
#define GBh_R(Rh,k)      (k)
#define GBi_R(Ri,p,vlen) Ri [p]
#define GBb_R(Rb,p)      1
#define GB_R_NVALS(e) int64_t e = R->nvals
#define GB_R_NHELD(e) GB_R_NVALS(e)
#define GB_Rp_TYPE uint64_t
#define GB_Rj_TYPE uint64_t
#define GB_Rj_SIGNED_TYPE int64_t
#define GB_Ri_TYPE uint64_t
#define GB_Ri_SIGNED_TYPE int64_t
#define GB_Rp_BITS 64
#define GB_Rj_BITS 64
#define GB_Ri_BITS 64

#include "include/GB_kernel_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__subref_bitmap__0808c6f
#define GB_jit_query  GB_jit__subref_bitmap__0808c6f_query
#endif
#include "template/GB_jit_kernel_subref_bitmap.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x9080eb43013af2e3 ;
    v [0] = 10 ; v [1] = 3 ; v [2] = 0 ;
    defn [0] = NULL ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
