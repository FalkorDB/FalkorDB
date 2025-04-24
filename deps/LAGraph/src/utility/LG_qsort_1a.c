//------------------------------------------------------------------------------
// LG_qsort_1a: sort an 1-by-n list of integers
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
    LG_lt_1 (A ## _0, a, B ## _0, b)

// argument list for calling a function
#define LG_arg(A)                       \
    A ## _0

// argument list for calling a function, with offset
#define LG_arg_offset(A,x)              \
    A ## _0 + (x)

// argument list for defining a function
#define LG_args(A)                      \
    int64_t *LG_RESTRICT A ## _0

// each entry has a single key
#define LG_K 1

// swap A [a] and A [b]
#define LG_swap(A,a,b)                                                        \
{                                                                             \
    int64_t t0 = A ## _0 [a] ; A ## _0 [a] = A ## _0 [b] ; A ## _0 [b] = t0 ; \
}

#define LG_partition LG_partition_1a
#define LG_quicksort LG_quicksort_1a

#include "LG_qsort_template.h"

void LG_qsort_1a    // sort array A of size 1-by-n
(
    int64_t *LG_RESTRICT A_0,       // size n array
    const int64_t n
)
{
    uint64_t seed = n ;
    LG_quicksort (LG_arg (A), n, &seed, NULL) ;
}
