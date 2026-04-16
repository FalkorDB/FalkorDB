//------------------------------------------------------------------------------
// GB_jit__reduce__12552.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.4.0, Timothy A. Davis, (c) 2017-2026,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// reduce: (min, uint16_t)

// monoid:
#define GB_Z_TYPE uint16_t
#define GB_ADD(z,x,y) z = (((x) < (y)) ? (x) : (y))
#ifdef  GB_CUDA_KERNEL
#define GB_UPDATE(z,y) z = (((z) < (y)) ? (z) : (y))
#else
#define GB_UPDATE(z,y) if ((z) > (y)) { z = y ; }
#endif
#define GB_DECLARE_IDENTITY(z) uint16_t z = UINT16_MAX
#define GB_DECLARE_IDENTITY_CONST(z) const uint16_t z = UINT16_MAX
#define GB_HAS_IDENTITY_BYTE 1
#define GB_IDENTITY_BYTE 0xff
#define GB_MONOID_IS_TERMINAL 1
#define GB_DECLARE_TERMINAL_CONST(zterminal) const uint16_t zterminal = 0
#define GB_TERMINAL_CONDITION(z,zterminal) ((z) == 0)
#define GB_IF_TERMINAL_BREAK(z,zterminal) if ((z) == 0) break
#define GB_IS_IMIN_MONOID 1
#define GB_Z_SIZE  2
#define GB_Z_NBITS 16
#define GB_Z_ATOMIC_BITS 16
#define GB_Z_HAS_ATOMIC_UPDATE 1
#define GB_Z_HAS_CUDA_ATOMIC_BUILTIN 1
#define GB_Z_CUDA_ATOMIC GB_cuda_atomic_min
#define GB_Z_CUDA_ATOMIC_TYPE uint32_t
#define GB_GETA_AND_UPDATE(z,Ax,p) GB_UPDATE (z, Ax [p])

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
#define GB_A_HAS_ZOMBIES 0
#define GB_A_ISO 0
#define GB_A_TYPE uint16_t
#define GB_A2TYPE uint16_t
#define GB_DECLAREA(a) uint16_t a
#define GB_GETA(a,Ax,p,iso) a = Ax [p]
#define GB_Ap_TYPE uint64_t
#define GB_Aj_TYPE uint64_t
#define GB_Aj_SIGNED_TYPE int64_t
#define GB_Ai_TYPE uint64_t
#define GB_Ai_SIGNED_TYPE int64_t
#define GB_Ap_BITS 64
#define GB_Aj_BITS 64
#define GB_Ai_BITS 64

// panel size for reduction:
#define GB_PANEL 16

#include "include/GB_monoid_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__reduce__12552
#define GB_jit_query  GB_jit__reduce__12552_query
#endif
#include "template/GB_jit_kernel_reduce.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0xbea62a5ced4a938f ;
    v [0] = 10 ; v [1] = 4 ; v [2] = 0 ;
    defn [0] = NULL ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
