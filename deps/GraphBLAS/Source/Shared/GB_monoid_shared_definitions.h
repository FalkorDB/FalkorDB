//------------------------------------------------------------------------------
// GB_monoid_shared_definitions.h: common macros for monoids
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_monoid_shared_definitions.h provides default definitions for all monoids,
// if the special cases have not been #define'd prior to #include'ing this
// file.

#include "GB_kernel_shared_definitions.h"

#ifndef GB_MONOID_SHARED_DEFINITIONS_H
#define GB_MONOID_SHARED_DEFINITIONS_H

//------------------------------------------------------------------------------
// special monoids
//------------------------------------------------------------------------------

// 1 if monoid is ANY
#ifndef GB_IS_ANY_MONOID
#define GB_IS_ANY_MONOID 0
#endif

// 1 if monoid is PLUS_FC32
#ifndef GB_IS_PLUS_FC32_MONOID
#define GB_IS_PLUS_FC32_MONOID 0
#endif

// 1 if monoid is PLUS_FC64
#ifndef GB_IS_PLUS_FC64_MONOID
#define GB_IS_PLUS_FC64_MONOID 0
#endif

// 1 if monoid is MIN for signed or unsigned integers
#ifndef GB_IS_IMIN_MONOID
#define GB_IS_IMIN_MONOID 0
#endif

// 1 if monoid is MAX for signed or unsigned integers
#ifndef GB_IS_IMAX_MONOID
#define GB_IS_IMAX_MONOID 0
#endif

// 1 if monoid is MIN for float or double
#ifndef GB_IS_FMIN_MONOID
#define GB_IS_FMIN_MONOID 0
#endif

// 1 if monoid is MAX for float or double
#ifndef GB_IS_FMAX_MONOID
#define GB_IS_FMAX_MONOID 0
#endif

//------------------------------------------------------------------------------
// monoid simd reduction
//------------------------------------------------------------------------------

// This macro expands into one of the following, or is empty:

//      #pragma omp simd reduction(+:z)
//      #pragma omp simd reduction(*:z)
//      #pragma omp simd reduction(^:z)
//      ...

// by default, no simd vectorization reduction #pragma
#ifndef GB_PRAGMA_SIMD_REDUCTION_MONOID
#define GB_PRAGMA_SIMD_REDUCTION_MONOID(z)
#endif

//------------------------------------------------------------------------------
// monoid atomic updates
//------------------------------------------------------------------------------

// if not specified, the monoid cannot be done via a single atomic operation.
// The update must instead be done inside a critical section.

#ifndef GB_Z_HAS_ATOMIC_UPDATE
#define GB_Z_HAS_ATOMIC_UPDATE 0
#endif

#ifndef GB_Z_HAS_OMP_ATOMIC_UPDATE
#define GB_Z_HAS_OMP_ATOMIC_UPDATE 0
#endif

#ifndef GB_Z_HAS_CUDA_ATOMIC_BUILTIN
#define GB_Z_HAS_CUDA_ATOMIC_BUILTIN 0
#endif

#ifndef GB_Z_HAS_CUDA_ATOMIC_USER
#define GB_Z_HAS_CUDA_ATOMIC_USER 0
#endif

#ifdef GB_CUDA_KERNEL
#if GB_Z_HAS_CUDA_ATOMIC_USER
static __device__ __inline__
void GB_cuda_atomic_user (void *pz, GB_Z_TYPE t)
{
    GB_Z_CUDA_ATOMIC_TYPE *p = (GB_Z_CUDA_ATOMIC_TYPE *) pz ;
    GB_Z_CUDA_ATOMIC_TYPE assumed ;
    GB_Z_CUDA_ATOMIC_TYPE old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // apply the pun to get the old value in GB_Z_TYPE
        GB_Z_TYPE zin = GB_PUN (GB_Z_TYPE, assumed) ;
        // compute the new value
        GB_Z_TYPE z ;
        GB_ADD (z, zin, t) ;
        // modify it atomically:
        old = atomicCAS (p, assumed, GB_PUN (GB_Z_CUDA_ATOMIC_TYPE, z)) ;
    }
    while (assumed != old) ;
}
#endif
#endif

//------------------------------------------------------------------------------
// monoid identity & terminal value and conditions, and handling ztype overflow
//------------------------------------------------------------------------------

// by default, monoid has no terminal value
#ifndef GB_DECLARE_TERMINAL_CONST
#define GB_DECLARE_TERMINAL_CONST(zterminal)
#endif

// by default, identity value is not a single repeated byte
#ifndef GB_HAS_IDENTITY_BYTE
#define GB_HAS_IDENTITY_BYTE 0
#endif
#ifndef GB_IDENTITY_BYTE
#define GB_IDENTITY_BYTE (none)
#endif

#if GB_IS_ANY_MONOID

    // by default, the ANY monoid is terminal
    #ifndef GB_MONOID_IS_TERMINAL
    #define GB_MONOID_IS_TERMINAL 1
    #endif
    #ifndef GB_TERMINAL_CONDITION
    #define GB_TERMINAL_CONDITION(z,zterminal) 1
    #endif
    #ifndef GB_IF_TERMINAL_BREAK
    #define GB_IF_TERMINAL_BREAK(z,zterminal) break
    #endif

    // ignore overflow since no numerical values computed
    #ifndef GB_Z_IGNORE_OVERFLOW
    #define GB_Z_IGNORE_OVERFLOW 1
    #endif

#else

    // monoids are not terminal unless explicitly declared otherwise
    #ifndef GB_MONOID_IS_TERMINAL
    #define GB_MONOID_IS_TERMINAL 0
    #endif
    #ifndef GB_TERMINAL_CONDITION
    #define GB_TERMINAL_CONDITION(z,zterminal) 0
    #endif
    #ifndef GB_IF_TERMINAL_BREAK
    #define GB_IF_TERMINAL_BREAK(z,zterminal)
    #endif

    // default, do not ignore overflow when replacing z+z+...+z with n*z.
    #ifndef GB_Z_IGNORE_OVERFLOW
    #define GB_Z_IGNORE_OVERFLOW 0
    #endif

#endif
#endif

