//------------------------------------------------------------------------------
// GB_msort_1: sort a 1-by-n list of integers
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A parallel mergesort of an array of 1-by-n integers.  Each key
// is a 32-bit or 64-bit unsigned integer.

#include "sort/GB_sort.h"

//------------------------------------------------------------------------------
// GB_msort_1_32
//------------------------------------------------------------------------------

#define GB_A0_t uint32_t
#define GB_msort_1_binary_search      GB_msort_1_binary_search_32
#define GB_msort_1_create_merge_tasks GB_msort_1_create_merge_tasks_32
#define GB_msort_1_merge              GB_msort_1_merge_32
#define GB_msort_1_method             GB_msort_1_32
#define GB_qsort_1_method             GB_qsort_1_32

#include "sort/factory/GB_msort_1_template.c"

//------------------------------------------------------------------------------
// GB_msort_1_64
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_msort_1_binary_search
#undef  GB_msort_1_create_merge_tasks
#undef  GB_msort_1_merge
#undef  GB_msort_1_method
#undef  GB_qsort_1_method

#define GB_A0_t uint64_t
#define GB_msort_1_binary_search      GB_msort_1_binary_search_64
#define GB_msort_1_create_merge_tasks GB_msort_1_create_merge_tasks_64
#define GB_msort_1_merge              GB_msort_1_merge_64
#define GB_msort_1_method             GB_msort_1_64
#define GB_qsort_1_method             GB_qsort_1_64

#include "sort/factory/GB_msort_1_template.c"

//------------------------------------------------------------------------------
// GB_msort_1
//------------------------------------------------------------------------------

GrB_Info GB_msort_1     // sort array A of size 1-by-n
(
    void *restrict A_0,         // size n array
    bool A0_is_32,              // if true: A_0 is uint32, else uint64
    const int64_t n,
    int nthreads_max            // max # of threads to use
)
{

    //--------------------------------------------------------------------------
    // handle small problems with a single thread
    //--------------------------------------------------------------------------

    int nthreads = GB_nthreads (n, GB_MSORT_BASECASE, nthreads_max) ;

#if 0
    // HACK: to test GB_bitonic:
    if (A0_is_32)
    {
        return (GB_bitonic (A_0, n, nthreads)) ;
    }
#endif

    if (nthreads <= 1 || n <= GB_MSORT_BASECASE)
    { 
        // sequential quicksort
        GB_qsort_1 (A_0, A0_is_32, n) ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // call the type-specific GB_msort_1 method
    //--------------------------------------------------------------------------

    if (A0_is_32)
    { 
        return (GB_msort_1_32 (A_0, n, nthreads)) ;
    }
    else
    { 
        return (GB_msort_1_64 (A_0, n, nthreads)) ;
    }
}

