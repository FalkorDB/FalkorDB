//------------------------------------------------------------------------------
// GB_mxm_shared_definitions.h: common macros for A*B kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_mxm_shared_definitions.h provides default definitions for all semirings,
// if the special cases have not been #define'd prior to #include'ing this
// file.  This file is shared by generic, pre-generated, and both CPU and CUDA
// JIT kernels.

#include "GB_monoid_shared_definitions.h"

#ifndef GB_MXM_SHARED_DEFINITIONS_H
#define GB_MXM_SHARED_DEFINITIONS_H

//------------------------------------------------------------------------------
// special semirings
//------------------------------------------------------------------------------

// 1 for the symbolic ANY_PAIR semiring
#ifndef GB_IS_ANY_PAIR_SEMIRING
#define GB_IS_ANY_PAIR_SEMIRING 0
#endif

// 1 for PLUS_PAIR semirings (integer, float, and double; not bool or complex)
#ifndef GB_IS_PLUS_PAIR_REAL_SEMIRING
#define GB_IS_PLUS_PAIR_REAL_SEMIRING 0
#endif

// 1 for LXOR_PAIR_BOOL
#ifndef GB_IS_LXOR_PAIR_SEMIRING
#define GB_IS_LXOR_PAIR_SEMIRING 0
#endif

// 1 for PLUS_PAIR_INT8 and PLUS_PAIR_UINT8
#ifndef GB_IS_PLUS_PAIR_8_SEMIRING
#define GB_IS_PLUS_PAIR_8_SEMIRING 0
#endif

// 1 for PLUS_PAIR_INT16 and PLUS_PAIR_UINT16
#ifndef GB_IS_PLUS_PAIR_16_SEMIRING
#define GB_IS_PLUS_PAIR_16_SEMIRING 0
#endif

// 1 for PLUS_PAIR_INT32 and PLUS_PAIR_UINT32
#ifndef GB_IS_PLUS_PAIR_32_SEMIRING
#define GB_IS_PLUS_PAIR_32_SEMIRING 0
#endif

// 1 for PLUS_PAIR_(INT64, UINT64, FP32, and FP64)
#ifndef GB_IS_PLUS_PAIR_BIG_SEMIRING
#define GB_IS_PLUS_PAIR_BIG_SEMIRING 0
#endif

// 1 for PLUS_PAIR_FC32
// #ifndef GB_IS_PLUS_PAIR_FC32_SEMIRING
// #define GB_IS_PLUS_PAIR_FC32_SEMIRING 0
// #endif

// 1 for PLUS_PAIR_FC64
// #ifndef GB_IS_PLUS_PAIR_FC64_SEMIRING
// #define GB_IS_PLUS_PAIR_FC64_SEMIRING 0
// #endif

// 1 for MIN_FIRSTJ
#ifndef GB_IS_MIN_FIRSTJ_SEMIRING
#define GB_IS_MIN_FIRSTJ_SEMIRING 0
#endif

// 1 for MAX_FIRSTJ
#ifndef GB_IS_MAX_FIRSTJ_SEMIRING
#define GB_IS_MAX_FIRSTJ_SEMIRING 0
#endif

// 1 if the semiring has an AVX512 or AVX2 implementation
#ifndef GB_SEMIRING_HAS_AVX_IMPLEMENTATION
#define GB_SEMIRING_HAS_AVX_IMPLEMENTATION 0
#endif

//------------------------------------------------------------------------------
// special multiply operators
//------------------------------------------------------------------------------

// 1 if the multiply operator is PAIR
#ifndef GB_IS_PAIR_MULTIPLIER
#define GB_IS_PAIR_MULTIPLIER 0
#endif

// 1 for the FIRSTI1, FIRSTJ1, SECONDI1, or SECONDJ1 multiply operators
#ifndef GB_OFFSET
#define GB_OFFSET 0
#endif

// 1 for the FIRSTI or FIRSTI1 multiply operator
#ifndef GB_IS_FIRSTI_MULTIPLIER
#define GB_IS_FIRSTI_MULTIPLIER 0
#endif

// 1 for the FIRSTJ, FIRSTJ1, SECONDI, or SECONDI1 multiply operator
#ifndef GB_IS_FIRSTJ_MULTIPLIER
#define GB_IS_FIRSTJ_MULTIPLIER 0
#endif

// 1 for the SECONDJ or SECONDJ1 multiply operator
#ifndef GB_IS_SECONDJ_MULTIPLIER
#define GB_IS_SECONDJ_MULTIPLIER 0
#endif

// 1 if values of A not accessed
#ifndef GB_A_IS_PATTERN
#define GB_A_IS_PATTERN 0
#endif

// 1 if values of B not accessed
#ifndef GB_B_IS_PATTERN
#define GB_B_IS_PATTERN 0
#endif

//------------------------------------------------------------------------------
// numerical operations and assignments
//------------------------------------------------------------------------------

#if GB_IS_ANY_PAIR_SEMIRING

    //--------------------------------------------------------------------------
    // ANY_PAIR semiring: no values are accessed
    //--------------------------------------------------------------------------

    // declare a scalar of ztype
    #ifndef GB_CIJ_DECLARE
    #define GB_CIJ_DECLARE(cij)
    #endif

    // Cx [p] = t
    #ifndef GB_CIJ_WRITE
    #define GB_CIJ_WRITE(p,t)
    #endif

    // Hx [i] = t
    #ifndef GB_HX_WRITE
    #define GB_HX_WRITE(i,t)
    #endif

    // Cx [p] = Hx [i]
    #ifndef GB_CIJ_GATHER
    #define GB_CIJ_GATHER(p,i)
    #endif

    // C(i,j) += t
    #ifndef GB_CIJ_UPDATE
    #define GB_CIJ_UPDATE(p,t)
    #endif

    // Cx [p] += Hx [i]
    #ifndef GB_CIJ_GATHER_UPDATE
    #define GB_CIJ_GATHER_UPDATE(p,i)
    #endif

    // Hx [i] += t
    #ifndef GB_HX_UPDATE
    #define GB_HX_UPDATE(i,t)
    #endif

    // Cx [p:p+len-1] = Hx [i:i+len-1]
    #ifndef GB_CIJ_MEMCPY
    #define GB_CIJ_MEMCPY(p,i,len)
    #endif

    // rest of the PAIR operator
    #ifndef GB_PAIR_ONE
    #define GB_PAIR_ONE 1
    #endif

#else

    //--------------------------------------------------------------------------
    // all pre-generated and JIT kernels
    //--------------------------------------------------------------------------

    // These definitions require explicit types to be used, not GB_void.
    // Generic methods using GB_void for all types, memcpy, and function
    // pointers for all computations must #define these macros first.

    // declare a scalar of ztype
    #ifndef GB_CIJ_DECLARE
    #define GB_CIJ_DECLARE(cij) GB_Z_TYPE cij
    #endif

    // Cx [p] = t
    #ifndef GB_CIJ_WRITE
    #define GB_CIJ_WRITE(p,t) Cx [p] = t
    #endif

    // Hx [i] = t
    #ifndef GB_HX_WRITE
    #define GB_HX_WRITE(i,t) Hx [i] = t
    #endif

    // Cx [p] = Hx [i]
    #ifndef GB_CIJ_GATHER
    #define GB_CIJ_GATHER(p,i) Cx [p] = Hx [i]
    #endif

    // C(i,j) += t
    #ifndef GB_CIJ_UPDATE
    #define GB_CIJ_UPDATE(p,t) GB_UPDATE (Cx [p], t)
    #endif

    // Hx [i] += t
    #ifndef GB_HX_UPDATE
    #define GB_HX_UPDATE(i,t) GB_UPDATE (Hx [i], t)
    #endif

    // Cx [p] += Hx [i]
    #ifndef GB_CIJ_GATHER_UPDATE
    #define GB_CIJ_GATHER_UPDATE(p,i) GB_UPDATE (Cx [p], Hx [i])
    #endif

    // Cx [p:p+len-1] = Hx [i:i+len-1]
    #ifndef GB_CIJ_MEMCPY
    #define GB_CIJ_MEMCPY(p,i,len) \
        memcpy (Cx +(p), Hx +(i), (len) * sizeof (GB_C_TYPE))
    #endif

    // rest of the PAIR operator
    #ifndef GB_PAIR_ONE
    #define GB_PAIR_ONE ((GB_Z_TYPE) 1)
    #endif

#endif
#endif
