//------------------------------------------------------------------------------
// GB_qsort_2: sort a 2-by-n list of integers, using A[0:1][ ] as the key
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Each entry consists of two 32-bit or 64-bit unsigned integers.

#include "sort/GB_sort.h"

// returns true if A [a] < B [b]
#define GB_lt(A,a,B,b)                  \
    GB_lt_2 (A ## _0, A ## _1, a, B ## _0, B ## _1, b)

// argument list for calling a function
#define GB_arg(A)                       \
    A ## _0, A ## _1

// argument list for calling a function, with offset
#define GB_arg_offset(A,x)              \
    A ## _0 + (x), A ## _1 + (x)

// argument list for defining a function
#define GB_args(A)                      \
    GB_A0_t *restrict A ## _0,          \
    GB_A1_t *restrict A ## _1

// each entry has a 2-integer key
#define GB_K 2

// swap A [a] and A [b]
#define GB_swap(A,a,b)                                                        \
{                                                                             \
    GB_A0_t t0 = A ## _0 [a] ; A ## _0 [a] = A ## _0 [b] ; A ## _0 [b] = t0 ; \
    GB_A1_t t1 = A ## _1 [a] ; A ## _1 [a] = A ## _1 [b] ; A ## _1 [b] = t1 ; \
}

//------------------------------------------------------------------------------
// GB_qsort_2_32_32
//------------------------------------------------------------------------------

#define GB_A0_t uint32_t
#define GB_A1_t uint32_t
#define GB_partition GB_partition_2_32_32
#define GB_quicksort GB_quicksort_2_32_32

#include "sort/template/GB_qsort_template.c"

void GB_qsort_2_32_32   // sort A of size 2-by-n, A0: 32bit, A1: 32bit
(
    GB_A0_t *restrict A_0,      // size n array
    GB_A1_t *restrict A_1,      // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_2_32_64
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_A1_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint32_t
#define GB_A1_t uint64_t
#define GB_partition GB_partition_2_32_64
#define GB_quicksort GB_quicksort_2_32_64

#include "sort/template/GB_qsort_template.c"

void GB_qsort_2_32_64   // sort A of size 2-by-n, A0: 32bit, A1: 64bit
(
    GB_A0_t *restrict A_0,      // size n array
    GB_A1_t *restrict A_1,      // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_2_64_32
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_A1_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint64_t
#define GB_A1_t uint32_t
#define GB_partition GB_partition_2_64_32
#define GB_quicksort GB_quicksort_2_64_32

#include "sort/template/GB_qsort_template.c"

void GB_qsort_2_64_32   // sort A of size 2-by-n, A0: 64bit, A1: 32bit
(
    GB_A0_t *restrict A_0,      // size n array
    GB_A1_t *restrict A_1,      // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_2_64_64
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_A1_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint64_t
#define GB_A1_t uint64_t
#define GB_partition GB_partition_2_64_64
#define GB_quicksort GB_quicksort_2_64_64

#include "sort/template/GB_qsort_template.c"

void GB_qsort_2_64_64   // sort A of size 2-by-n, A0: 64bit, A1: 64bit
(
    GB_A0_t *restrict A_0,      // size n array
    GB_A1_t *restrict A_1,      // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_2: for 32/64-bit cases
//------------------------------------------------------------------------------

void GB_qsort_2     // sort array A of size 2-by-n, using 2 keys (A [0:1][])
(
    void *restrict A_0,         // size n array
    bool A0_is_32,              // if true: A_0 is uint32, false: uint64
    void *restrict A_1,         // size n array
    bool A1_is_32,              // if true: A_1 is uint32, false: uint64
    const int64_t n
)
{ 
    if (A0_is_32)
    {
        if (A1_is_32)
        { 
            // A0: uint32_t, A1: uint32_t
            GB_qsort_2_32_32 (A_0, A_1, n) ;
        }
        else
        { 
            // A0: uint32_t, A1: uint64_t
            GB_qsort_2_32_64 (A_0, A_1, n) ;
        }
    }
    else
    {
        if (A1_is_32)
        { 
            // A0: uint64_t, A1: uint32_t
            GB_qsort_2_64_32 (A_0, A_1, n) ;
        }
        else
        { 
            // A0: uint64_t, A1: uint64_t
            GB_qsort_2_64_64 (A_0, A_1, n) ;
        }
    }
}

