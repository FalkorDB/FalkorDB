//------------------------------------------------------------------------------
// GB_jit__emult_08__e3f10a999099904.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.0.2, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: (bitor, uint64_t)

// binary operator types:
#define GB_Z_TYPE uint64_t
#define GB_X_TYPE uint64_t
#define GB_Y_TYPE uint64_t

// binary operator:
#define GB_BINOP(z,x,y,i,j) z = ((x) | (y))
#define GB_COPY_A_to_C(Cx,pC,Ax,pA,A_iso)
#define GB_COPY_B_to_C(Cx,pC,Bx,pB,B_iso)

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
#define GB_C_TYPE uint64_t
#define GB_PUTC(c,Cx,p) Cx [p] = c
#define GB_Cp_TYPE uint32_t
#define GB_Cj_TYPE uint32_t
#define GB_Ci_TYPE uint32_t
#define GB_Ci_SIGNED_TYPE int32_t
#define GB_Cp_BITS 32
#define GB_Cj_BITS 32
#define GB_Ci_BITS 32
#define GB_EWISEOP(Cx,p,aij,bij,i,j) GB_BINOP (Cx [p], aij, bij, i, j)

// M matrix: none
#define GB_M_TYPE void
#define GB_MCAST(Mx,p,msize) 1
#define GB_MASK_STRUCT 1
#define GB_MASK_COMP   0
#define GB_NO_MASK     1
#define GB_Mp_TYPE uint64_t
#define GB_Mj_TYPE uint64_t
#define GB_Mi_TYPE uint64_t
#define GB_Mi_SIGNED_TYPE int64_t
#define GB_Mp_BITS 64
#define GB_Mj_BITS 64
#define GB_Mi_BITS 64

// A matrix: sparse
#define GB_A_IS_HYPER  0
#define GB_A_IS_SPARSE 1
#define GB_A_IS_BITMAP 0
#define GB_A_IS_FULL   0
#define GBp_A(Ap,k,vlen) Ap [k]
#define GBh_A(Ah,k)      (k)
#define GBi_A(Ai,p,vlen) Ai [p]
#define GBb_A(Ab,p)      1
#define GB_A_NVALS(e) int64_t e = A->nvals
#define GB_A_NHELD(e) GB_A_NVALS(e)
#define GB_A_ISO 0
#define GB_A_TYPE uint64_t
#define GB_A2TYPE uint64_t
#define GB_DECLAREA(a) uint64_t a
#define GB_GETA(a,Ax,p,iso) a = Ax [p]
#define GB_Ap_TYPE uint32_t
#define GB_Aj_TYPE uint32_t
#define GB_Ai_TYPE uint32_t
#define GB_Ai_SIGNED_TYPE int32_t
#define GB_Ap_BITS 32
#define GB_Aj_BITS 32
#define GB_Ai_BITS 32

// B matrix: hypersparse
#define GB_B_IS_HYPER  1
#define GB_B_IS_SPARSE 0
#define GB_B_IS_BITMAP 0
#define GB_B_IS_FULL   0
#define GBp_B(Bp,k,vlen) Bp [k]
#define GBh_B(Bh,k)      Bh [k]
#define GBi_B(Bi,p,vlen) Bi [p]
#define GBb_B(Bb,p)      1
#define GB_B_NVALS(e) int64_t e = B->nvals
#define GB_B_NHELD(e) GB_B_NVALS(e)
#define GB_B_ISO 1
#define GB_B_TYPE uint64_t
#define GB_B2TYPE uint64_t
#define GB_DECLAREB(b) uint64_t b
#define GB_GETB(b,Bx,p,iso) b = Bx [0]
#define GB_Bp_TYPE uint32_t
#define GB_Bj_TYPE uint32_t
#define GB_Bi_TYPE uint32_t
#define GB_Bi_SIGNED_TYPE int32_t
#define GB_Bp_BITS 32
#define GB_Bj_BITS 32
#define GB_Bi_BITS 32

#include "include/GB_ewise_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__emult_08__e3f10a999099904
#define GB_jit_query  GB_jit__emult_08__e3f10a999099904_query
#endif
#include "template/GB_jit_kernel_emult_08.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x5e7485e5685b5a76 ;
    v [0] = 10 ; v [1] = 0 ; v [2] = 2 ;
    defn [0] = NULL ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
