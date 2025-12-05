//------------------------------------------------------------------------------
// GB_jit__masker_phase1__dff33044.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.3.0, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// masker: 

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
#define GB_R_ISO 0
#define GB_Rp_TYPE uint32_t
#define GB_Rj_TYPE uint32_t
#define GB_Rj_SIGNED_TYPE int32_t
#define GB_Ri_TYPE uint64_t
#define GB_Ri_SIGNED_TYPE int64_t
#define GB_Rp_BITS 32
#define GB_Rj_BITS 32
#define GB_Ri_BITS 64

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
#define GB_C_ISO 1
#define GB_Cp_TYPE uint32_t
#define GB_Cj_TYPE uint32_t
#define GB_Cj_SIGNED_TYPE int32_t
#define GB_Ci_TYPE uint32_t
#define GB_Ci_SIGNED_TYPE int32_t
#define GB_Cp_BITS 32
#define GB_Cj_BITS 32
#define GB_Ci_BITS 32

// M matrix: sparse
#define GB_M_IS_HYPER  0
#define GB_M_IS_SPARSE 1
#define GB_M_IS_BITMAP 0
#define GB_M_IS_FULL   0
#define GBp_M(Mp,k,vlen) Mp [k]
#define GBh_M(Mh,k)      (k)
#define GBi_M(Mi,p,vlen) Mi [p]
#define GBb_M(Mb,p)      1
// structural mask (complemented):
#define GB_M_TYPE void
#define GB_MCAST(Mx,p,msize) 1
#define GB_MASK_STRUCT 1
#define GB_MASK_COMP   1
#define GB_NO_MASK     0
#define GB_M_NVALS(e) int64_t e = M->nvals
#define GB_M_NHELD(e) GB_M_NVALS(e)
#define GB_Mp_TYPE uint32_t
#define GB_Mj_TYPE uint32_t
#define GB_Mj_SIGNED_TYPE int32_t
#define GB_Mi_TYPE uint32_t
#define GB_Mi_SIGNED_TYPE int32_t
#define GB_Mp_BITS 32
#define GB_Mj_BITS 32
#define GB_Mi_BITS 32

// Z matrix: hypersparse
#define GB_Z_IS_HYPER  1
#define GB_Z_IS_SPARSE 0
#define GB_Z_IS_BITMAP 0
#define GB_Z_IS_FULL   0
#define GBp_Z(Zp,k,vlen) Zp [k]
#define GBh_Z(Zh,k)      Zh [k]
#define GBi_Z(Zi,p,vlen) Zi [p]
#define GBb_Z(Zb,p)      1
#define GB_Z_NVALS(e) int64_t e = Z->nvals
#define GB_Z_NHELD(e) GB_Z_NVALS(e)
#define GB_Z_ISO 1
#define GB_Zp_TYPE uint32_t
#define GB_Zj_TYPE uint32_t
#define GB_Zj_SIGNED_TYPE int32_t
#define GB_Zi_TYPE uint32_t
#define GB_Zi_SIGNED_TYPE int32_t
#define GB_Zp_BITS 32
#define GB_Zj_BITS 32
#define GB_Zi_BITS 32

#include "include/GB_masker_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__masker_phase1__dff33044
#define GB_jit_query  GB_jit__masker_phase1__dff33044_query
#endif
#include "template/GB_jit_kernel_masker_phase1.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x8a0ee6fdf5022b9c ;
    v [0] = 10 ; v [1] = 3 ; v [2] = 0 ;
    defn [0] = NULL ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
