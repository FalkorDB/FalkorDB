//------------------------------------------------------------------------------
// LG_qsort_3: sort a 3-by-n list of integers, using A[0:2][] as the key
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

#include "LG_internal.h"

// returns true if A [a] < B [b]
#define LG_lt(A,a,B,b)                  \
    LG_lt_3 (A ## _0, A ## _1, A ## _2, a, B ## _0, B ## _1, B ## _2, b)

// argument list for calling a function
#define LG_arg(A)                       \
    A ## _0, A ## _1, A ## _2

// argument list for calling a function, with offset
#define LG_arg_offset(A,x)              \
    A ## _0 + (x), A ## _1 + (x), A ## _2 + (x)

// argument list for defining a function
#define LG_args(A)                      \
    int64_t *LG_RESTRICT A ## _0,       \
    int64_t *LG_RESTRICT A ## _1,       \
    int64_t *LG_RESTRICT A ## _2

// each entry has a 3-integer key
#define LG_K 3

// swap A [a] and A [b]
#define LG_swap(A,a,b)                                                        \
{                                                                             \
    int64_t t0 = A ## _0 [a] ; A ## _0 [a] = A ## _0 [b] ; A ## _0 [b] = t0 ; \
    int64_t t1 = A ## _1 [a] ; A ## _1 [a] = A ## _1 [b] ; A ## _1 [b] = t1 ; \
    int64_t t2 = A ## _2 [a] ; A ## _2 [a] = A ## _2 [b] ; A ## _2 [b] = t2 ; \
}

#define LG_partition LG_partition_3
#define LG_quicksort LG_quicksort_3

#include "LG_qsort_template.h"

void LG_qsort_3     // sort array A of size 3-by-n, using 3 keys (A [0:2][])
(
    int64_t *LG_RESTRICT A_0,       // size n array
    int64_t *LG_RESTRICT A_1,       // size n array
    int64_t *LG_RESTRICT A_2,       // size n array
    const int64_t n
)
{
    uint64_t seed = n ;
    LG_quicksort (LG_arg (A), n, &seed, NULL) ;
}
