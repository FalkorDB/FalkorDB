//------------------------------------------------------------------------------
// GB_qsort_1: sort an 1-by-n list of integers
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Each entry is a single 32-bit or 64-bit unsigned integer.

#include "sort/GB_sort.h"

// returns true if A [a] < B [b]
#define GB_lt(A,a,B,b)                  \
    GB_lt_1 (A ## _0, a, B ## _0, b)

// argument list for calling a function
#define GB_arg(A)                       \
    A ## _0

// argument list for calling a function, with offset
#define GB_arg_offset(A,x)              \
    A ## _0 + (x)

// argument list for defining a function
#define GB_args(A)                      \
    GB_A0_t *restrict A ## _0

// each entry has a single key
#define GB_K 1

// swap A [a] and A [b]
#define GB_swap(A,a,b)                                                        \
{                                                                             \
    GB_A0_t t0 = A ## _0 [a] ; A ## _0 [a] = A ## _0 [b] ; A ## _0 [b] = t0 ; \
}

//------------------------------------------------------------------------------
// GB_qsort_1_32
//------------------------------------------------------------------------------

#define GB_A0_t uint32_t
#define GB_partition GB_partition_1_32
#define GB_quicksort GB_quicksort_1_32

#include "sort/template/GB_qsort_template.c"

void GB_qsort_1_32
(
    GB_A0_t *restrict A_0,      // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (A_0, n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_1_64
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint64_t
#define GB_partition GB_partition_1_64
#define GB_quicksort GB_quicksort_1_64

#include "sort/template/GB_qsort_template.c"

void GB_qsort_1_64
(
    GB_A0_t *restrict A_0,      // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (A_0, n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_1: sort an array A of size 1-by-n, 32 or 64 bit
//------------------------------------------------------------------------------

void GB_qsort_1
(
    void *restrict A_0,         // size n array
    bool A0_is_32,              // if true: A_0 is 32-bit; else 64-bit
    const int64_t n
)
{
    if (A0_is_32)
    { 
        GB_qsort_1_32 ((uint32_t *) A_0, n) ;
    }
    else
    { 
        GB_qsort_1_64 ((uint64_t *) A_0, n) ;
    }
}

