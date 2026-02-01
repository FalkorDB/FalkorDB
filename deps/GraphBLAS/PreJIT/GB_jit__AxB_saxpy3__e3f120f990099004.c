//------------------------------------------------------------------------------
// GB_jit__AxB_saxpy3__e3f120f990099004.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.3.1, Timothy A. Davis, (c) 2017-2026,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// semiring: (any, first, uint64_t)

// monoid:
#define GB_Z_TYPE uint64_t
#define GB_ADD(z,x,y) z = y
#define GB_UPDATE(z,y) z = y
#define GB_DECLARE_IDENTITY(z) uint64_t z = 0
#define GB_DECLARE_IDENTITY_CONST(z) const uint64_t z = 0
#define GB_HAS_IDENTITY_BYTE 1
#define GB_IDENTITY_BYTE 0x00
#define GB_IS_ANY_MONOID 1
#define GB_Z_SIZE  8
#define GB_Z_NBITS 64
#define GB_Z_ATOMIC_BITS 64
#define GB_Z_HAS_ATOMIC_UPDATE 1
#define GB_Z_HAS_OMP_ATOMIC_UPDATE 1
#define GB_Z_HAS_CUDA_ATOMIC_BUILTIN 1
#define GB_Z_CUDA_ATOMIC GB_cuda_atomic_write
#define GB_Z_CUDA_ATOMIC_TYPE uint64_t

// multiplicative operator:
#define GB_X_TYPE uint64_t
#define GB_Y_TYPE void
#define GB_MULT(z,x,y,i,k,j) z = x

// multiply-add operator:
#define GB_MULTADD(z,x,y,i,k,j) z = x

// special cases:

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
#define GB_Cj_SIGNED_TYPE int32_t
#define GB_Ci_TYPE uint32_t
#define GB_Ci_SIGNED_TYPE int32_t
#define GB_Cp_BITS 32
#define GB_Cj_BITS 32
#define GB_Ci_BITS 32

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
#define GB_Aj_SIGNED_TYPE int32_t
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
#define GB_B_IS_PATTERN 1
#define GB_B_TYPE void
#define GB_B2TYPE void
#define GB_DECLAREB(b)
#define GB_GETB(b,Bx,p,iso)
#define GB_Bp_TYPE uint32_t
#define GB_Bj_TYPE uint32_t
#define GB_Bj_SIGNED_TYPE int32_t
#define GB_Bi_TYPE uint32_t
#define GB_Bi_SIGNED_TYPE int32_t
#define GB_Bp_BITS 32
#define GB_Bj_BITS 32
#define GB_Bi_BITS 32

#include "include/GB_mxm_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__AxB_saxpy3__e3f120f990099004
#define GB_jit_query  GB_jit__AxB_saxpy3__e3f120f990099004_query
#endif
#include "template/GB_jit_kernel_AxB_saxpy3.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0xbfafbbebb77a627d ;
    v [0] = 10 ; v [1] = 3 ; v [2] = 1 ;
    defn [0] = NULL ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
