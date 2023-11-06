//------------------------------------------------------------------------------
// GB_qsort_3: sort a 3-by-n list of integers, using A[0:2][] as the key
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_sort.h"

// returns true if A [a] < B [b]
#define GB_lt(A,a,B,b)                  \
    GB_lt_3 (A ## _0, A ## _1, A ## _2, a, B ## _0, B ## _1, B ## _2, b)

// argument list for calling a function
#define GB_arg(A)                       \
    A ## _0, A ## _1, A ## _2

// argument list for calling a function, with offset
#define GB_arg_offset(A,x)              \
    A ## _0 + (x), A ## _1 + (x), A ## _2 + (x)

// argument list for defining a function
#define GB_args(A)                      \
    int64_t *restrict A ## _0,          \
    int64_t *restrict A ## _1,          \
    int64_t *restrict A ## _2

// each entry has a 3-integer key
#define GB_K 3

// swap A [a] and A [b]
#define GB_swap(A,a,b)                                                        \
{                                                                             \
    int64_t t0 = A ## _0 [a] ; A ## _0 [a] = A ## _0 [b] ; A ## _0 [b] = t0 ; \
    int64_t t1 = A ## _1 [a] ; A ## _1 [a] = A ## _1 [b] ; A ## _1 [b] = t1 ; \
    int64_t t2 = A ## _2 [a] ; A ## _2 [a] = A ## _2 [b] ; A ## _2 [b] = t2 ; \
}

#define GB_partition GB_partition_3
#define GB_quicksort GB_quicksort_3

#include "GB_qsort_template.c"

void GB_qsort_3     // sort array A of size 3-by-n, using 3 keys (A [0:2][])
(
    int64_t *restrict A_0,      // size n array
    int64_t *restrict A_1,      // size n array
    int64_t *restrict A_2,      // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

