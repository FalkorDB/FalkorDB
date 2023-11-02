//------------------------------------------------------------------------------
// GB_kernel_shared_definitions.h: definitions for all methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This header is #include'd just before using any templates for any method:
// pre-generated kernel, CPU or GPU JIT, or generic.

#include "GB_unused.h"
#include "GB_complex.h"

#ifndef GB_KERNEL_SHARED_DEFINITIONS_H
#define GB_KERNEL_SHARED_DEFINITIONS_H

//------------------------------------------------------------------------------
// atomic compare/exchange for the GB_Z_TYPE data type
//------------------------------------------------------------------------------

// 1 if ztype is complex
#ifndef GB_Z_IS_COMPLEX
#define GB_Z_IS_COMPLEX 0
#endif

#if defined ( GB_Z_ATOMIC_BITS ) && !defined ( GB_GENERIC ) && !defined ( GB_CUDA_KERNEL )

    //--------------------------------------------------------------------------
    // atomic compared/exchange (0, 1, 2, 4, or 8 byte types)
    //--------------------------------------------------------------------------

    // pre-generated kernels can use these operations on built-in types.  CPU
    // JIT kernels can use them for user-defined types of the right size.

    #if ( GB_Z_ATOMIC_BITS == 0 )

        // no atomic compare/exchange needed (any_pair semiring)
        #define GB_Z_ATOMIC_COMPARE_EXCHANGE(target, expected, desired)

    #elif ( GB_Z_ATOMIC_BITS == 8 )

        // atomic compare/exchange for int8_t, uint8_t
        #define GB_Z_ATOMIC_COMPARE_EXCHANGE(target, expected, desired) \
                GB_ATOMIC_COMPARE_EXCHANGE_8(target, expected, desired)

    #elif ( GB_Z_ATOMIC_BITS == 16 )

        // atomic compare/exchange for int16_t, uint16_t
        #define GB_Z_ATOMIC_COMPARE_EXCHANGE(target, expected, desired) \
               GB_ATOMIC_COMPARE_EXCHANGE_16(target, expected, desired)

    #elif ( GB_Z_ATOMIC_BITS == 32 )

        // atomic compare/exchange for int32_t, uint32_t, and float
        #define GB_Z_ATOMIC_COMPARE_EXCHANGE(target, expected, desired) \
               GB_ATOMIC_COMPARE_EXCHANGE_32(target, expected, desired)

    #else // ( GB_Z_ATOMIC_BITS == 64 )

        // atomic compare/exchange for int64_t, uint64_t, double,
        // and float complex
        #define GB_Z_ATOMIC_COMPARE_EXCHANGE(target, expected, desired) \
               GB_ATOMIC_COMPARE_EXCHANGE_64(target, expected, desired)

    #endif

    //--------------------------------------------------------------------------
    // atomic write (0, 1, 2, 4, or 8 byte types)
    //--------------------------------------------------------------------------

    // if GB_Z_HAS_ATOMIC_WRITE is true, then the Z type has an atomic
    // write, an atomic read, and an atomic compare/exchange
    #define GB_Z_HAS_ATOMIC_WRITE 1

    #if ( GB_Z_ATOMIC_BITS == 0 )

        // no atomic read/write needed (any_pair semiring)
        #define GB_Z_ATOMIC_READ(z,t)
        #define GB_Z_ATOMIC_WRITE(z,t)

    #elif defined ( GB_Z_ATOMIC_TYPE )

        // user-defined types of the right size can use atomic read/write.
        // float complex also uses this version.
        #define GB_Z_ATOMIC_READ(z,t)                                       \
        {                                                                   \
            GB_ATOMIC_READ                                                  \
            GB_PUN (GB_Z_ATOMIC_TYPE, z) = GB_PUN (GB_Z_ATOMIC_TYPE, t) ;   \
        }
        #define GB_Z_ATOMIC_WRITE(z,t)                                      \
        {                                                                   \
            GB_ATOMIC_WRITE                                                 \
            GB_PUN (GB_Z_ATOMIC_TYPE, z) = GB_PUN (GB_Z_ATOMIC_TYPE, t) ;   \
        }

    #else

        // built-in types of size 1, 2, 4, or 8 bytes
        #define GB_Z_ATOMIC_READ(z,t)                                       \
        {                                                                   \
            GB_ATOMIC_READ                                                  \
            (z) = (t) ;                                                     \
        }
        #define GB_Z_ATOMIC_WRITE(z,t)                                      \
        {                                                                   \
            GB_ATOMIC_WRITE                                                 \
            (z) = (t) ;                                                     \
        }

    #endif

#else

    //--------------------------------------------------------------------------
    // no atomic compare/exchange or atomic write
    //--------------------------------------------------------------------------

    // Attempting to use the atomic compare/exchange will generate an
    // intentional compile-time error.

    // Generic kernels cannot use a single atomic compare/exchange method
    // determined at compile time since their size is run-time dependent.

    // CUDA kernels must use the atomics defined in GB_cuda_atomics.cuh,
    // not these methods.

    // If GB_Z_ATOMIC_BITS is not #define'd, then the kernel does not have a
    // GB_Z_TYPE, or it's not the correct size to use an atomic write or
    // atomic compare/exchange.

    #define GB_Z_HAS_ATOMIC_WRITE 0
    #define GB_Z_ATOMIC_COMPARE_EXCHANGE(target, expected, desired) none
    #define GB_Z_ATOMIC_WRITE(z,t) none

#endif
#endif

