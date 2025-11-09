//------------------------------------------------------------------------------
// GB_qsort_3: sort a 3-by-n list of integers, using A[0:2][] as the key
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Each entry consists of three 32-bit or 64-bit unsigned integers.

#include "sort/GB_sort.h"

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
    GB_A0_t *restrict A ## _0,          \
    GB_A1_t *restrict A ## _1,          \
    GB_A2_t *restrict A ## _2

// each entry has a 3-integer key
#define GB_K 3

// swap A [a] and A [b]
#define GB_swap(A,a,b)                                                        \
{                                                                             \
    GB_A0_t t0 = A ## _0 [a] ; A ## _0 [a] = A ## _0 [b] ; A ## _0 [b] = t0 ; \
    GB_A1_t t1 = A ## _1 [a] ; A ## _1 [a] = A ## _1 [b] ; A ## _1 [b] = t1 ; \
    GB_A2_t t2 = A ## _2 [a] ; A ## _2 [a] = A ## _2 [b] ; A ## _2 [b] = t2 ; \
}

//------------------------------------------------------------------------------
// GB_qsort_3_32_32_32
//------------------------------------------------------------------------------

#define GB_A0_t uint32_t
#define GB_A1_t uint32_t
#define GB_A2_t uint32_t
#define GB_partition GB_partition_3_32_32_32
#define GB_quicksort GB_quicksort_3_32_32_32

#include "sort/template/GB_qsort_template.c"

void GB_qsort_3_32_32_32 // sort A of size 3-by-n, A0: 32bit, A1: 32bit, A2: 32
(
    GB_A0_t *restrict A_0,      // size n array
    GB_A1_t *restrict A_1,      // size n array
    GB_A2_t *restrict A_2,      // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_3_32_32_64
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_A1_t
#undef  GB_A2_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint32_t
#define GB_A1_t uint32_t
#define GB_A2_t uint64_t
#define GB_partition GB_partition_3_32_32_64
#define GB_quicksort GB_quicksort_3_32_32_64

#include "sort/template/GB_qsort_template.c"

void GB_qsort_3_32_32_64 // sort A of size 3-by-n, A0: 32bit, A1: 32bit, A2: 64
(
    GB_A0_t *restrict A_0,      // size n array
    GB_A1_t *restrict A_1,      // size n array
    GB_A2_t *restrict A_2,      // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_3_32_64_32
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_A1_t
#undef  GB_A2_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint32_t
#define GB_A1_t uint64_t
#define GB_A2_t uint32_t
#define GB_partition GB_partition_3_32_64_32
#define GB_quicksort GB_quicksort_3_32_64_32

#include "sort/template/GB_qsort_template.c"

void GB_qsort_3_32_64_32 // sort A of size 3-by-n, A0: 32bit, A1: 64, A2: 32
(
    GB_A0_t *restrict A_0,      // size n array
    GB_A1_t *restrict A_1,      // size n array
    GB_A2_t *restrict A_2,      // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_3_32_64_64
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_A1_t
#undef  GB_A2_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint32_t
#define GB_A1_t uint64_t
#define GB_A2_t uint64_t
#define GB_partition GB_partition_3_32_64_64
#define GB_quicksort GB_quicksort_3_32_64_64

#include "sort/template/GB_qsort_template.c"

void GB_qsort_3_32_64_64 // sort A of size 3-by-n, A0: 32bit, A1: 64, A2: 64
(
    GB_A0_t *restrict A_0,      // size n array
    GB_A1_t *restrict A_1,      // size n array
    GB_A2_t *restrict A_2,      // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_3_64_32_32
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_A1_t
#undef  GB_A2_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint64_t
#define GB_A1_t uint32_t
#define GB_A2_t uint32_t
#define GB_partition GB_partition_3_64_32_32
#define GB_quicksort GB_quicksort_3_64_32_32

#include "sort/template/GB_qsort_template.c"

void GB_qsort_3_64_32_32 // sort A of size 3-by-n, A0: 64bit, A1: 32bit, A2: 32
(
    GB_A0_t *restrict A_0,      // size n array
    GB_A1_t *restrict A_1,      // size n array
    GB_A2_t *restrict A_2,      // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_3_64_32_64
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_A1_t
#undef  GB_A2_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint64_t
#define GB_A1_t uint32_t
#define GB_A2_t uint64_t
#define GB_partition GB_partition_3_64_32_64
#define GB_quicksort GB_quicksort_3_64_32_64

#include "sort/template/GB_qsort_template.c"

void GB_qsort_3_64_32_64 // sort A of size 3-by-n, A0: 64bit, A1: 32bit, A2: 64
(
    GB_A0_t *restrict A_0,      // size n array
    GB_A1_t *restrict A_1,      // size n array
    GB_A2_t *restrict A_2,      // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_3_64_64_32
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_A1_t
#undef  GB_A2_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint64_t
#define GB_A1_t uint64_t
#define GB_A2_t uint32_t
#define GB_partition GB_partition_3_64_64_32
#define GB_quicksort GB_quicksort_3_64_64_32

#include "sort/template/GB_qsort_template.c"

void GB_qsort_3_64_64_32 // sort A of size 3-by-n, A0: 64bit, A1: 64, A2: 32
(
    GB_A0_t *restrict A_0,      // size n array
    GB_A1_t *restrict A_1,      // size n array
    GB_A2_t *restrict A_2,      // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_3_64_64_64
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_A1_t
#undef  GB_A2_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint64_t
#define GB_A1_t uint64_t
#define GB_A2_t uint64_t
#define GB_partition GB_partition_3_64_64_64
#define GB_quicksort GB_quicksort_3_64_64_64

#include "sort/template/GB_qsort_template.c"

void GB_qsort_3_64_64_64 // sort A of size 3-by-n, A0: 64bit, A1: 64, A2: 64
(
    GB_A0_t *restrict A_0,      // size n array
    GB_A1_t *restrict A_1,      // size n array
    GB_A2_t *restrict A_2,      // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_3: for 32/64-bit cases
//------------------------------------------------------------------------------

void GB_qsort_3     // sort array A of size 3-by-n, using 3 keys (A [0:2][])
(
    void *restrict A_0,         // size n array
    bool A0_is_32,              // if true: A_0 is uint32, false: uint64
    void *restrict A_1,         // size n array
    bool A1_is_32,              // if true: A_1 is uint32, false: uint64
    void *restrict A_2,         // size n array
    bool A2_is_32,              // if true: A_1 is uint32, false: uint64
    const int64_t n
)
{ 
    if (A0_is_32)
    {
        if (A1_is_32)
        {
            if (A2_is_32)
            { 
                // A0: uint32_t, A1: uint32_t, A2: uint32_t
                GB_qsort_3_32_32_32 (A_0, A_1, A_2, n) ;
            }
            else
            { 
                // A0: uint32_t, A1: uint32_t, A2: uint64_t
                GB_qsort_3_32_32_64 (A_0, A_1, A_2, n) ;
            }
        }
        else
        {
            if (A2_is_32)
            { 
                // A0: uint32_t, A1: uint64_t, A2: uint32_t
                GB_qsort_3_32_64_32 (A_0, A_1, A_2, n) ;
            }
            else
            { 
                // A0: uint32_t, A1: uint64_t, A2: uint64_t
                GB_qsort_3_32_64_64 (A_0, A_1, A_2, n) ;
            }
        }
    }
    else
    {
        if (A1_is_32)
        {
            if (A2_is_32)
            { 
                // A0: uint64_t, A1: uint32_t, A2: uint32_t
                GB_qsort_3_64_32_32 (A_0, A_1, A_2, n) ;
            }
            else
            { 
                // A0: uint64_t, A1: uint32_t, A2: uint64_t
                GB_qsort_3_64_32_64 (A_0, A_1, A_2, n) ;
            }
        }
        else
        {
            if (A2_is_32)
            { 
                // A0: uint64_t, A1: uint64_t, A2: uint32_t
                GB_qsort_3_64_64_32 (A_0, A_1, A_2, n) ;
            }
            else
            { 
                // A0: uint64_t, A1: uint64_t, A2: uint64_t
                GB_qsort_3_64_64_64 (A_0, A_1, A_2, n) ;
            }
        }
    }
}

