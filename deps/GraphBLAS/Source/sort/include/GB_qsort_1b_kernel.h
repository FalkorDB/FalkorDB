//------------------------------------------------------------------------------
// GB_qsort_1b_kernel: sort a 2-by-n list, using A [0][ ] as the sort key
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This inline function is used in JIT/PreJIT kernels only.  The JIT kernel
// #define's GB_A0_t as uint32_t or uint64_t, and GB_A1_t as the type of A_1.

//------------------------------------------------------------------------------

#ifndef GB_QSORT_1B_KERNEL_H
#define GB_QSORT_1B_KERNEL_H

#include "include/GB_sort_kernels.h"

#define GB_partition GB_partition_1b_kernel
#define GB_quicksort GB_quicksort_1b_kernel

// returns true if A [a] < B [b]
#define GB_lt(A,a,B,b) GB_lt_1 (A ## _0, a, B ## _0, b)

// each entry has a single key
#define GB_K 1

// argument list for calling a function
#define GB_arg(A) A ## _0, A ## _1

// argument list for calling a function, with offset
#define GB_arg_offset(A,x) A ## _0 + (x), A ## _1 + (x)

// argument list for defining a function
#define GB_args(A) GB_A0_t *restrict A ## _0, GB_A1_t *restrict A ## _1

// swap A [a] and A [b]
#define GB_swap(A,a,b)                  \
{                                       \
    GB_A0_t t0 = A ## _0 [a] ; A ## _0 [a] = A ## _0 [b] ; A ## _0 [b] = t0 ; \
    GB_A1_t t1 = A ## _1 [a] ; A ## _1 [a] = A ## _1 [b] ; A ## _1 [b] = t1 ; \
}

#include "template/GB_qsort_template.c"

//------------------------------------------------------------------------------
// GB_qsort_1b_kernel: with A_0, A_1 having templatized types GB_A0_t, GB_A1_t
//------------------------------------------------------------------------------

static inline void GB_qsort_1b_kernel
(
    GB_A0_t *restrict A_0,      // size n array
    GB_A1_t *restrict A_1,      // size n array
    const int64_t n
)
{
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

#endif

