//------------------------------------------------------------------------------
// GB_qsort_1: sort an 1-by-n list of integers
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_sort.h"

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
    int64_t *restrict A ## _0

// each entry has a single key
#define GB_K 1

// swap A [a] and A [b]
#define GB_swap(A,a,b)                                                        \
{                                                                             \
    int64_t t0 = A ## _0 [a] ; A ## _0 [a] = A ## _0 [b] ; A ## _0 [b] = t0 ; \
}

#define GB_partition GB_partition_1
#define GB_quicksort GB_quicksort_1

#include "GB_qsort_template.c"

GB_CALLBACK_QSORT_1_PROTO (GB_qsort_1)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

