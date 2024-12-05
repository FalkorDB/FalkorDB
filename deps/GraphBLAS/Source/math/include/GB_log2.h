//------------------------------------------------------------------------------
// GB_log2.h: integer log2 and check if power of 2
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_LOG2_H
#define GB_LOG2_H

// # of bits in an unsigned long long, normally 64
#define GB_64 (8 * sizeof (unsigned long long))

// floor and ceiling of the log2 of an integer.
#if GB_COMPILER_MSC
// Microsoft Visual Studio does not have the __builtin_clzll
#define GB_CLZLL(k)   not defined, using log2 instead
#define GB_CEIL_LOG2(k)  ((uint64_t) (ceil  (log2 ((double) k))))
#define GB_FLOOR_LOG2(k) ((uint64_t) (floor (log2 ((double) k))))
#else
// see https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html
#define GB_CLZLL(k)   __builtin_clzll ((unsigned long long) (k))
#define GB_CEIL_LOG2(k)  ((uint64_t) ((k) < 2) ? 0 : (GB_64 - GB_CLZLL ((k)-1)))
#define GB_FLOOR_LOG2(k) ((uint64_t) ((k) < 2) ? 0 : (GB_64 - GB_CLZLL (k) - 1))
#endif

// GB_IS_POWER_OF_TWO(k) is true if the unsigned integer k is an exact power of
// two, or if k is zero.  This expression should not be used if k is negative.
#define GB_IS_POWER_OF_TWO(k) (((k) & ((k) - 1)) == 0)

#endif

