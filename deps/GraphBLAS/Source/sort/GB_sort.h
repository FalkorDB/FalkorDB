//------------------------------------------------------------------------------
// GB_sort.h: definitions for sorting functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// All of the GB_qsort_* functions are single-threaded, by design.  The
// GB_msort_* functions are parallel.  None of these sorting methods are
// guaranteed to be stable, but they are always used in GraphBLAS with unique
// keys, so they do not have to be stable.

#ifndef GB_SORT_H
#define GB_SORT_H

#include "GB.h"
#include "sort/include/GB_sort_kernels.h"
#define GB_MSORT_BASECASE (2*1024)

void GB_qsort_1b_32_generic // sort array A of size 2-by-n, using A0: 32 bit
(
    uint32_t *restrict A_0,     // size n array
    GB_void *restrict A_1,      // size n array
    const size_t xsize,         // size of entries in A_1
    const int64_t n
) ;

void GB_qsort_1b_64_generic // sort array A of size 2-by-n, using A0: 64 bit
(
    uint64_t *restrict A_0,     // size n array
    GB_void *restrict A_1,      // size n array
    const size_t xsize,         // size of entries in A_1
    const int64_t n
) ;

void GB_qsort_1b_32_size1  // GB_qsort_1b, A1 with sizeof = 1, A0: 32 bit
(
    uint32_t *restrict A_0,     // size n array
    uint8_t *restrict A_1,      // size n array
    const int64_t n
) ;

void GB_qsort_1b_64_size1  // GB_qsort_1b, A1 with sizeof = 1, A0: 64 bit
(
    uint64_t *restrict A_0,     // size n array
    uint8_t *restrict A_1,      // size n array
    const int64_t n
) ;

void GB_qsort_1b_32_size2  // GB_qsort_1b, A1 with sizeof = 2, A0: 32 bit
(
    uint32_t *restrict A_0,     // size n array
    uint16_t *restrict A_1,     // size n array
    const int64_t n
) ;

void GB_qsort_1b_64_size2  // GB_qsort_1b, A1 with sizeof = 2, A0: 64 bit
(
    uint64_t *restrict A_0,     // size n array
    uint16_t *restrict A_1,     // size n array
    const int64_t n
) ;

void GB_qsort_1b_32_size4  // GB_qsort_1b, A1 with sizeof = 4, A0: 32 bit
(
    uint32_t *restrict A_0,     // size n array
    uint32_t *restrict A_1,     // size n array
    const int64_t n
) ;

void GB_qsort_1b_64_size4  // GB_qsort_1b, A1 with sizeof = 4, A0: 64 bit
(
    uint64_t *restrict A_0,     // size n array
    uint32_t *restrict A_1,     // size n array
    const int64_t n
) ;

void GB_qsort_1b_32_size8  // GB_qsort_1b, A_1 with sizeof = 8, A0: 32 bit
(
    uint32_t *restrict A_0,     // size n array
    uint64_t *restrict A_1,     // size n array
    const int64_t n
) ;

void GB_qsort_1b_64_size8  // GB_qsort_1b, A_1 with sizeof = 8, A0: 64 bit
(
    uint64_t *restrict A_0,     // size n array
    uint64_t *restrict A_1,     // size n array
    const int64_t n
) ;

void GB_qsort_1b_32_size16 // GB_qsort_1b, A_1 with sizeof = 16, A0: 64 bit
(
    uint32_t *restrict A_0,     // size n array
    GB_blob16 *restrict A_1,    // size n array
    const int64_t n
) ;

void GB_qsort_1b_64_size16 // GB_qsort_1b, A_1 with sizeof = 16, A0: 64 bit
(
    uint64_t *restrict A_0,     // size n array
    GB_blob16 *restrict A_1,    // size n array
    const int64_t n
) ;

void GB_qsort_1_32
(
    uint32_t *restrict A_0,     // size n array
    const int64_t n
) ;

void GB_qsort_1_64
(
    uint64_t *restrict A_0,     // size n array
    const int64_t n
) ;

void GB_qsort_1
(
    void *restrict A_0,         // size n array
    bool A0_is_32,              // if true: A_0 is 32-bit; else 64-bit
    const int64_t n
) ;

void GB_qsort_2_64_64   // sort A of size 2-by-n, A0: 64bit, A1: 64bit
(
    uint64_t *restrict A_0,     // size n array
    uint64_t *restrict A_1,     // size n array
    const int64_t n
) ;

void GB_qsort_2_32_64   // sort A of size 2-by-n, A0: 32bit, A1: 64bit
(
    uint32_t *restrict A_0,     // size n array
    uint64_t *restrict A_1,     // size n array
    const int64_t n
) ;

void GB_qsort_2_64_32   // sort A of size 2-by-n, A0: 64bit, A1: 32bit
(
    uint64_t *restrict A_0,     // size n array
    uint32_t *restrict A_1,     // size n array
    const int64_t n
) ;

void GB_qsort_2_32_32   // sort A of size 2-by-n, A0: 32bit, A1: 32bit
(
    uint32_t *restrict A_0,     // size n array
    uint32_t *restrict A_1,     // size n array
    const int64_t n
) ;

void GB_qsort_2     // sort array A of size 2-by-n, using 2 keys (A [0:1][])
(
    void *restrict A_0,         // size n array
    bool A0_is_32,              // if true: A_0 is uint32, false: uint64
    void *restrict A_1,         // size n array
    bool A1_is_32,              // if true: A_1 is uint32, false: uint64
    const int64_t n
) ;

void GB_qsort_3_32_32_32 // sort A of size 3-by-n, A0: 32bit, A1: 32, A2: 32
(
    uint32_t *restrict A_0,     // size n array
    uint32_t *restrict A_1,     // size n array
    uint32_t *restrict A_2,     // size n array
    const int64_t n
) ;

void GB_qsort_3_32_32_64 // sort A of size 3-by-n, A0: 32bit, A1: 32, A2: 64
(
    uint32_t *restrict A_0,     // size n array
    uint32_t *restrict A_1,     // size n array
    uint64_t *restrict A_2,     // size n array
    const int64_t n
) ;

void GB_qsort_3_32_64_32 // sort A of size 3-by-n, A0: 32bit, A1: 64, A2: 32
(
    uint32_t *restrict A_0,     // size n array
    uint64_t *restrict A_1,     // size n array
    uint32_t *restrict A_2,     // size n array
    const int64_t n
) ;

void GB_qsort_3_32_64_64 // sort A of size 3-by-n, A0: 32bit, A1: 64, A2: 64
(
    uint32_t *restrict A_0,     // size n array
    uint64_t *restrict A_1,     // size n array
    uint64_t *restrict A_2,     // size n array
    const int64_t n
) ;

void GB_qsort_3_64_32_32 // sort A of size 3-by-n, A0: 64bit, A1: 32, A2: 32
(
    uint64_t *restrict A_0,     // size n array
    uint32_t *restrict A_1,     // size n array
    uint32_t *restrict A_2,     // size n array
    const int64_t n
) ;

void GB_qsort_3_64_32_64 // sort A of size 3-by-n, A0: 64bit, A1: 32, A2: 64
(
    uint64_t *restrict A_0,     // size n array
    uint32_t *restrict A_1,     // size n array
    uint64_t *restrict A_2,     // size n array
    const int64_t n
) ;

void GB_qsort_3_64_64_32 // sort A of size 3-by-n, A0: 64bit, A1: 64, A2: 32
(
    uint64_t *restrict A_0,     // size n array
    uint64_t *restrict A_1,     // size n array
    uint32_t *restrict A_2,     // size n array
    const int64_t n
) ;

void GB_qsort_3_64_64_64 // sort A of size 3-by-n, A0: 64bit, A1: 64, A2: 64
(
    uint64_t *restrict A_0,     // size n array
    uint64_t *restrict A_1,     // size n array
    uint64_t *restrict A_2,     // size n array
    const int64_t n
) ;

void GB_qsort_3     // sort array A of size 3-by-n, using 3 keys (A [0:2][])
(
    void *restrict A_0,         // size n array
    bool A0_is_32,              // if true: A_0 is uint32, false: uint64
    void *restrict A_1,         // size n array
    bool A1_is_32,              // if true: A_1 is uint32, false: uint64
    void *restrict A_2,         // size n array
    bool A2_is_32,              // if true: A_1 is uint32, false: uint64
    const int64_t n
) ;

GrB_Info GB_msort_1     // sort array A of size 1-by-n
(
    void *restrict A_0,         // size n array
    bool A0_is_32,              // if true: A_0 is uint32, else uint64
    const int64_t n,
    int nthreads_max            // max # of threads to use
) ;

GrB_Info GB_msort_2     // sort array A of size 2-by-n
(
    void *restrict A_0,         // size n array
    bool A0_is_32,              // if true: A_0 is uint32, else uint64
    void *restrict A_1,         // size n array
    bool A1_is_32,              // if true: A_1 is uint32, else uint64
    const int64_t n,
    int nthreads_max            // max # of threads to use
) ;

GrB_Info GB_msort_3     // sort array A of size 3-by-n
(
    void *restrict A_0,         // size n array
    bool A0_is_32,              // if true: A_0 is uint32, else uint64
    void *restrict A_1,         // size n array
    bool A1_is_32,              // if true: A_1 is uint32, else uint64
    void *restrict A_2,         // size n array
    bool A2_is_32,              // if true: A_2 is uint32, else uint64
    const int64_t n,
    int nthreads_max            // max # of threads to use
) ;

//------------------------------------------------------------------------------
// bitonic sort
//------------------------------------------------------------------------------

GrB_Info GB_bitonic
(
    int32_t *restrict A,    // array of size n
    int64_t n,              // n does not need to be a power of 2
    int nthreads
) ;

//------------------------------------------------------------------------------
// matrix sorting (for GxB_Matrix_sort and GxB_Vector_sort)
//------------------------------------------------------------------------------

GrB_Info GB_sort
(
    // output:
    GrB_Matrix C,               // matrix with sorted vectors on output
    GrB_Matrix P,               // matrix with permutations on output
    // input:
    GrB_BinaryOp op,            // comparator for the sort
    GrB_Matrix A,               // matrix to sort
    const bool A_transpose,     // false: sort each row, true: sort each column
    GB_Werk Werk
) ;

#endif

