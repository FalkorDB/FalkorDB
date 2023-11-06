//------------------------------------------------------------------------------
// GB_qsort_1b: sort a 2-by-n list, using A [0][ ] as the sort key
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_sort.h"

// returns true if A [a] < B [b]
#define GB_lt(A,a,B,b) GB_lt_1 (A ## _0, a, B ## _0, b)

// each entry has a single key
#define GB_K 1

//------------------------------------------------------------------------------
// GB_qsort_1b: generic method for any data type
//------------------------------------------------------------------------------

// argument list for calling a function
#define GB_arg(A)                       \
    A ## _0, A ## _1, xsize

// argument list for calling a function, with offset
#define GB_arg_offset(A,x)              \
    A ## _0 + (x), A ## _1 + (x)*xsize, xsize

// argument list for defining a function
#define GB_args(A)                      \
    int64_t *restrict A ## _0,       \
    GB_void *restrict A ## _1,       \
    size_t xsize

// swap A [a] and A [b]
#define GB_swap(A,a,b)                                                        \
{                                                                             \
    int64_t t0 = A ## _0 [a] ; A ## _0 [a] = A ## _0 [b] ; A ## _0 [b] = t0 ; \
    GB_void t1 [GB_VLA(xsize)] ;                                              \
    memcpy (t1, A ## _1 + (a)*xsize, xsize) ;                                 \
    memcpy (A ## _1 + (a)*xsize, A ## _1 + (b)*xsize, xsize) ;                \
    memcpy (A ## _1 + (b)*xsize, t1, xsize) ;                                 \
}

#define GB_partition GB_partition_1b
#define GB_quicksort GB_quicksort_1b

#include "GB_qsort_template.c"

void GB_qsort_1b    // sort array A of size 2-by-n, using 1 key (A [0][])
(
    int64_t *restrict A_0,       // size n array
    GB_void *restrict A_1,       // size n array
    const size_t xsize,          // size of entries in A_1
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_1b_size1:  quicksort with A_1 of type that has sizeof 1
//------------------------------------------------------------------------------

// for GrB_BOOL, GrB_INT8, GrB_UINT8, and user-defined types with sizeof(...)=1

#define A1_type uint8_t

// argument list for calling a function
#undef  GB_arg
#define GB_arg(A)                       \
    A ## _0, A ## _1

// argument list for calling a function, with offset
#undef  GB_arg_offset
#define GB_arg_offset(A,x)              \
    A ## _0 + (x), A ## _1 + (x)

// argument list for defining a function
#undef  GB_args
#define GB_args(A)                      \
    int64_t *restrict A ## _0,          \
    A1_type *restrict A ## _1           \

// swap A [a] and A [b]
#undef  GB_swap
#define GB_swap(A,a,b)                  \
{                                       \
    int64_t t0 = A ## _0 [a] ; A ## _0 [a] = A ## _0 [b] ; A ## _0 [b] = t0 ; \
    A1_type t1 = A ## _1 [a] ; A ## _1 [a] = A ## _1 [b] ; A ## _1 [b] = t1 ; \
}

#undef  GB_partition
#define GB_partition GB_partition_1b_size1
#undef  GB_quicksort
#define GB_quicksort GB_quicksort_1b_size1

#include "GB_qsort_template.c"

void GB_qsort_1b_size1  // GB_qsort_1b with A_1 with sizeof = 1
(
    int64_t *restrict A_0,       // size n array
    uint8_t *restrict A_1,       // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_1b_size2:  quicksort with A_1 of type that has sizeof 2
//------------------------------------------------------------------------------

// for GrB_INT16, GrB_UINT16, and user-defined types of sizeof(...) = 2

#undef  A1_type
#define A1_type uint16_t
#undef  GB_partition
#define GB_partition GB_partition_1b_size2
#undef  GB_quicksort
#define GB_quicksort GB_quicksort_1b_size2

#include "GB_qsort_template.c"

void GB_qsort_1b_size2  // GB_qsort_1b with A_1 with sizeof = 2
(
    int64_t *restrict A_0,       // size n array
    uint16_t *restrict A_1,      // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_1b_size4:  quicksort with A_1 of type that has sizeof 4
//------------------------------------------------------------------------------

// for GrB_INT32, GrB_UINT32, GrB_FP32, and user-defined types with
// sizeof(...) = 4.

#undef  A1_type
#define A1_type uint32_t
#undef  GB_partition
#define GB_partition GB_partition_1b_size4
#undef  GB_quicksort
#define GB_quicksort GB_quicksort_1b_size4

#include "GB_qsort_template.c"

void GB_qsort_1b_size4  // GB_qsort_1b with A_1 with sizeof = 4
(
    int64_t *restrict A_0,       // size n array
    uint32_t *restrict A_1,      // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_1b_size8:  quicksort with A_1 of type that has sizeof 8
//------------------------------------------------------------------------------

// for GrB_INT64, GrB_UINT64, GrB_FP64, GxB_FC32, and user-defined types
// with sizeof(...) = 8.

#undef  A1_type
#define A1_type uint64_t
#undef  GB_partition
#define GB_partition GB_partition_1b_size8
#undef  GB_quicksort
#define GB_quicksort GB_quicksort_1b_size8

#include "GB_qsort_template.c"

void GB_qsort_1b_size8  // GB_qsort_1b with A_1 with sizeof = 8
(
    int64_t *restrict A_0,       // size n array
    uint64_t *restrict A_1,      // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_1b_size16:  quicksort with A_1 of type that has sizeof 16
//------------------------------------------------------------------------------

// for GxB_FC64 and user-defined types with sizeof(...) = 16.

#undef  A1_type
#define A1_type GB_blob16
#undef  GB_partition
#define GB_partition GB_partition_1b_size16
#undef  GB_quicksort
#define GB_quicksort GB_quicksort_1b_size16

#include "GB_qsort_template.c"

void GB_qsort_1b_size16 // GB_qsort_1b with A_1 with sizeof = 16
(
    int64_t *restrict A_0,       // size n array
    GB_blob16 *restrict A_1,     // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

