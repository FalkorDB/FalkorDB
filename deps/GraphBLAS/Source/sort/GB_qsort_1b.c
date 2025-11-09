//------------------------------------------------------------------------------
// GB_qsort_1b: sort a 2-by-n list, using A [0][ ] as the sort key
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "sort/GB_sort.h"

// returns true if A [a] < B [b]
#define GB_lt(A,a,B,b) GB_lt_1 (A ## _0, a, B ## _0, b)

// Each entry has a single key: a 32-bit or 64-bit unsigned integer
#define GB_K 1

//------------------------------------------------------------------------------
// GB_qsort_1b_32_generic: generic method for any data type, A0 is 32bit
//------------------------------------------------------------------------------

#define GB_A0_t uint32_t
#define GB_A1_t GB_void
#define GB_partition GB_partition_1b_32_generic
#define GB_quicksort GB_quicksort_1b_32_generic

// argument list for calling a function
#define GB_arg(A)                       \
    A ## _0, A ## _1, xsize

// argument list for calling a function, with offset
#define GB_arg_offset(A,x)              \
    A ## _0 + (x), A ## _1 + (x)*xsize, xsize

// argument list for defining a function
#define GB_args(A)                      \
    GB_A0_t *restrict A ## _0,          \
    GB_A1_t *restrict A ## _1,          \
    size_t xsize

// swap A [a] and A [b]
#define GB_swap(A,a,b)                                                        \
{                                                                             \
    GB_A0_t t0 = A ## _0 [a] ; A ## _0 [a] = A ## _0 [b] ; A ## _0 [b] = t0 ; \
    GB_A1_t t1 [GB_VLA(xsize)] ;                                              \
    memcpy (t1, A ## _1 + (a)*xsize, xsize) ;                                 \
    memcpy (A ## _1 + (a)*xsize, A ## _1 + (b)*xsize, xsize) ;                \
    memcpy (A ## _1 + (b)*xsize, t1, xsize) ;                                 \
}

#include "sort/template/GB_qsort_template.c"

void GB_qsort_1b_32_generic // sort array A of size 2-by-n, using A0: 32
(
    GB_A0_t *restrict A_0,      // size n array
    GB_A1_t *restrict A_1,      // size n array
    const size_t xsize,         // size of entries in A_1
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_1b_64_generic: generic method for any data type, A0 is 64bit
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint64_t
#define GB_partition GB_partition_1b_64_generic
#define GB_quicksort GB_quicksort_1b_64_generic

#include "sort/template/GB_qsort_template.c"

void GB_qsort_1b_64_generic // sort array A of size 2-by-n, using A0: 64
(
    GB_A0_t *restrict A_0,      // size n array
    GB_A1_t *restrict A_1,      // size n array
    const size_t xsize,         // size of entries in A_1
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

//------------------------------------------------------------------------------
// GB_qsort_1b_32_size1:  quicksort, A_1 of type that has sizeof 1, A0: 32bit
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_A1_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint32_t
#define GB_A1_t uint8_t
#define GB_partition GB_partition_1b_32_size1
#define GB_quicksort GB_quicksort_1b_32_size1

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
    GB_A0_t *restrict A ## _0,          \
    GB_A1_t *restrict A ## _1           \

// swap A [a] and A [b]
#undef  GB_swap
#define GB_swap(A,a,b)                  \
{                                       \
    GB_A0_t t0 = A ## _0 [a] ; A ## _0 [a] = A ## _0 [b] ; A ## _0 [b] = t0 ; \
    GB_A1_t t1 = A ## _1 [a] ; A ## _1 [a] = A ## _1 [b] ; A ## _1 [b] = t1 ; \
}

#include "sort/template/GB_qsort_template.c"

void GB_qsort_1b_32_size1  // GB_qsort_1b, A_1 with sizeof = 1, A0: 32 bit
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
// GB_qsort_1b_64_size1:  quicksort, A_1 of type that has sizeof 1, A0: 64bit
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint64_t
#define GB_partition GB_partition_1b_64_size1
#define GB_quicksort GB_quicksort_1b_64_size1

#include "sort/template/GB_qsort_template.c"

void GB_qsort_1b_64_size1  // GB_qsort_1b with A_1 with sizeof = 1, A0: 64
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
// GB_qsort_1b_32_size2:  quicksort: A_1 of type with sizeof 2, A0: 32 bit
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_A1_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint32_t
#define GB_A1_t uint16_t
#define GB_partition GB_partition_1b_32_size2
#define GB_quicksort GB_quicksort_1b_32_size2

#include "sort/template/GB_qsort_template.c"

void GB_qsort_1b_32_size2  // GB_qsort_1b, A_1 with sizeof = 2, A0: 32 bit
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
// GB_qsort_1b_64_size2:  quicksort: A_1 of type with sizeof 2, A0: 64 bit
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint64_t
#define GB_partition GB_partition_1b_64_size2
#define GB_quicksort GB_quicksort_1b_64_size2

#include "sort/template/GB_qsort_template.c"

void GB_qsort_1b_64_size2  // GB_qsort_1b, A_1 with sizeof = 2, A0: 64 bit
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
// GB_qsort_1b_32_size4:  quicksort, A_1 of type that has sizeof 4, A0: 32 bit
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_A1_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint32_t
#define GB_A1_t uint32_t
#define GB_partition GB_partition_1b_32_size4
#define GB_quicksort GB_quicksort_1b_32_size4

#include "sort/template/GB_qsort_template.c"

void GB_qsort_1b_32_size4  // GB_qsort_1b A_1 with sizeof = 4, A0: 32 bit
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
// GB_qsort_1b_64_size4:  quicksort, A_1 of type that has sizeof 4, A0: 64 bit
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint64_t
#define GB_partition GB_partition_1b_64_size4
#define GB_quicksort GB_quicksort_1b_64_size4

#include "sort/template/GB_qsort_template.c"

void GB_qsort_1b_64_size4  // GB_qsort_1b A_1 with sizeof = 4, A0: 64 bit
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
// GB_qsort_1b_32_size8:  quicksort, A_1 of type that has sizeof 8, A0: 32 bit
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_A1_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint32_t
#define GB_A1_t uint64_t
#define GB_partition GB_partition_1b_32_size8
#define GB_quicksort GB_quicksort_1b_32_size8

#include "sort/template/GB_qsort_template.c"

void GB_qsort_1b_32_size8  // GB_qsort_1b, A_1 with sizeof = 8, A0: 32 bit
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
// GB_qsort_1b_64_size8:  quicksort, A_1 of type that has sizeof 8, A0: 64 bit
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint64_t
#define GB_partition GB_partition_1b_64_size8
#define GB_quicksort GB_quicksort_1b_64_size8

#include "sort/template/GB_qsort_template.c"

void GB_qsort_1b_64_size8  // GB_qsort_1b, A_1 with sizeof = 8, A0: 64 bit
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
// GB_qsort_1b_32_size16:  quicksort, A_1 of type that has sizeof 16, A0: 32 bit
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_A1_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint32_t
#define GB_A1_t GB_blob16
#define GB_partition GB_partition_1b_32_size16
#define GB_quicksort GB_quicksort_1b_32_size16

#include "sort/template/GB_qsort_template.c"

void GB_qsort_1b_32_size16 // GB_qsort_1b, A_1 with sizeof = 16, A0: 32 bit
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
// GB_qsort_1b_64_size16:  quicksort, A_1 of type that has sizeof 16, A0: 64 bit
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_partition
#undef  GB_quicksort

#define GB_A0_t uint64_t
#define GB_partition GB_partition_1b_64_size16
#define GB_quicksort GB_quicksort_1b_64_size16

#include "sort/template/GB_qsort_template.c"

void GB_qsort_1b_64_size16 // GB_qsort_1b, A_1 with sizeof = 16, A0: 64 bit
(
    GB_A0_t *restrict A_0,      // size n array
    GB_A1_t *restrict A_1,      // size n array
    const int64_t n
)
{ 
    uint64_t seed = n ;
    GB_quicksort (GB_arg (A), n, &seed) ;
}

