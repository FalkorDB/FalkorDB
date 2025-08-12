//------------------------------------------------------------------------------
// GB_jit__reduce__171b2.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.0.2, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// reduce: (and, bool)

// monoid:
#define GB_Z_TYPE bool
#define GB_ADD(z,x,y) z = ((x) && (y))
#define GB_UPDATE(z,y) z &= y
#define GB_DECLARE_IDENTITY(z) bool z = true
#define GB_DECLARE_IDENTITY_CONST(z) const bool z = true
#define GB_HAS_IDENTITY_BYTE 1
#define GB_IDENTITY_BYTE 0x01
#define GB_MONOID_IS_TERMINAL 1
#define GB_DECLARE_TERMINAL_CONST(zterminal) const bool zterminal = false
#define GB_TERMINAL_CONDITION(z,zterminal) ((z) == false)
#define GB_IF_TERMINAL_BREAK(z,zterminal) if ((z) == false) break
#define GB_Z_SIZE  1
#define GB_Z_NBITS 8
#define GB_Z_ATOMIC_BITS 8
#define GB_Z_HAS_ATOMIC_UPDATE 1
#define GB_Z_HAS_OMP_ATOMIC_UPDATE (!GB_COMPILER_MSC)
#define GB_Z_HAS_CUDA_ATOMIC_BUILTIN 1
#define GB_Z_CUDA_ATOMIC GB_cuda_atomic_band
#define GB_Z_CUDA_ATOMIC_TYPE uint32_t
#define GB_GETA_AND_UPDATE(z,Ax,p) \
{                             \
    /* z += (ztype) Ax [p] */ \
    GB_DECLAREA (aij) ;       \
    GB_GETA (aij, Ax, p, ) ;  \
    GB_UPDATE (z, aij) ;      \
}

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
#define GB_A_TYPE double
#define GB_A2TYPE bool
#define GB_DECLAREA(a) bool a
#define GB_GETA(a,Ax,p,iso) a = ((Ax [p]) != 0)
#define GB_Ap_TYPE uint64_t
#define GB_Aj_TYPE uint64_t
#define GB_Ai_TYPE uint64_t
#define GB_Ai_SIGNED_TYPE int64_t
#define GB_Ap_BITS 64
#define GB_Aj_BITS 64
#define GB_Ai_BITS 64

// panel size for reduction:
#define GB_PANEL 8

#include "include/GB_monoid_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__reduce__171b2
#define GB_jit_query  GB_jit__reduce__171b2_query
#endif
#include "template/GB_jit_kernel_reduce.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x975f265e1ec6e115 ;
    v [0] = 10 ; v [1] = 0 ; v [2] = 2 ;
    defn [0] = NULL ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
