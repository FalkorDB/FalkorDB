//------------------------------------------------------------------------------
// GB_jit__reduce__12bb9.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.4.0, Timothy A. Davis, (c) 2017-2026,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// reduce: (min, double)

// monoid:
#define GB_Z_TYPE double
#define GB_ADD(z,x,y) z = fmin (x,y)
#ifdef  GB_CUDA_KERNEL
#define GB_UPDATE(z,y) z = fmin (z,y)
#else
#define GB_UPDATE(z,y) if (!isnan (y) && !islessequal (z,y)) { z = y ; }
#endif
#define GB_DECLARE_IDENTITY(z) double z = INFINITY
#define GB_DECLARE_IDENTITY_CONST(z) const double z = INFINITY
#define GB_IS_FMIN_MONOID 1
#define GB_Z_IGNORE_OVERFLOW 1
#define GB_Z_SIZE  8
#define GB_Z_NBITS 64
#define GB_Z_ATOMIC_BITS 64
#define GB_Z_HAS_ATOMIC_UPDATE 1
#define GB_Z_HAS_CUDA_ATOMIC_BUILTIN 1
#define GB_Z_CUDA_ATOMIC GB_cuda_atomic_min
#define GB_Z_CUDA_ATOMIC_TYPE double
#define GB_GETA_AND_UPDATE(z,Ax,p) GB_UPDATE (z, Ax [p])

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
#define GB_A_HAS_ZOMBIES 0
#define GB_A_ISO 0
#define GB_A_TYPE double
#define GB_A2TYPE double
#define GB_DECLAREA(a) double a
#define GB_GETA(a,Ax,p,iso) a = Ax [p]
#define GB_Ap_TYPE uint64_t
#define GB_Aj_TYPE uint64_t
#define GB_Aj_SIGNED_TYPE int64_t
#define GB_Ai_TYPE uint32_t
#define GB_Ai_SIGNED_TYPE int32_t
#define GB_Ap_BITS 64
#define GB_Aj_BITS 64
#define GB_Ai_BITS 32

// panel size for reduction:
#define GB_PANEL 16

#include "include/GB_monoid_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__reduce__12bb9
#define GB_jit_query  GB_jit__reduce__12bb9_query
#endif
#include "template/GB_jit_kernel_reduce.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0xbb597c24615d9cb2 ;
    v [0] = 10 ; v [1] = 4 ; v [2] = 0 ;
    defn [0] = NULL ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
