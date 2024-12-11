//------------------------------------------------------------------------------
// GB_sort.h: definitions for sorting functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// All of the GB_qsort_* functions are single-threaded, by design.  The
// GB_msort_* functions are parallel.  None of these sorting methods are
// guaranteed to be stable, but they are always used in GraphBLAS with unique
// keys.

#ifndef GB_SORT_H
#define GB_SORT_H

#include "GB.h"
#include "sort/include/GB_sort_kernels.h"

void GB_qsort_1b    // sort array A of size 2-by-n, using 1 key (A [0][])
(
    int64_t *restrict A_0,      // size n array
    GB_void *restrict A_1,      // size n array
    const size_t xsize,         // size of entries in A_1
    const int64_t n
) ;

void GB_qsort_1b_size1  // GB_qsort_1b with A1 with sizeof = 1
(
    int64_t *restrict A_0,       // size n array
    uint8_t *restrict A_1,       // size n array
    const int64_t n
) ;

void GB_qsort_1b_size2  // GB_qsort_1b with A1 with sizeof = 2
(
    int64_t *restrict A_0,       // size n array
    uint16_t *restrict A_1,      // size n array
    const int64_t n
) ;

void GB_qsort_1b_size4  // GB_qsort_1b with A1 with sizeof = 4
(
    int64_t *restrict A_0,       // size n array
    uint32_t *restrict A_1,      // size n array
    const int64_t n
) ;

void GB_qsort_1b_size8  // GB_qsort_1b with A_1 with sizeof = 8
(
    int64_t *restrict A_0,       // size n array
    uint64_t *restrict A_1,      // size n array
    const int64_t n
) ;

void GB_qsort_1b_size16 // GB_qsort_1b with A_1 with sizeof = 16
(
    int64_t *restrict A_0,       // size n array
    GB_blob16 *restrict A_1,     // size n array
    const int64_t n
) ;

void GB_qsort_2     // sort array A of size 2-by-n, using 2 keys (A [0:1][])
(
    int64_t *restrict A_0,      // size n array
    int64_t *restrict A_1,      // size n array
    const int64_t n
) ;

void GB_qsort_3     // sort array A of size 3-by-n, using 3 keys (A [0:2][])
(
    int64_t *restrict A_0,      // size n array
    int64_t *restrict A_1,      // size n array
    int64_t *restrict A_2,      // size n array
    const int64_t n
) ;

GrB_Info GB_msort_1     // sort array A of size 1-by-n
(
    int64_t *restrict A_0,   // size n array
    const int64_t n,
    int nthreads                // # of threads to use
) ;

GrB_Info GB_msort_2    // sort array A of size 2-by-n, using 2 keys (A [0:1][])
(
    int64_t *restrict A_0,   // size n array
    int64_t *restrict A_1,   // size n array
    const int64_t n,
    int nthreads                // # of threads to use
) ;

GrB_Info GB_msort_3    // sort array A of size 3-by-n, using 3 keys (A [0:2][])
(
    int64_t *restrict A_0,   // size n array
    int64_t *restrict A_1,   // size n array
    int64_t *restrict A_2,   // size n array
    const int64_t n,
    int nthreads                // # of threads to use
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


