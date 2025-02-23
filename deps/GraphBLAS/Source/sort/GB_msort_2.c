//------------------------------------------------------------------------------
// GB_msort_2: sort a 2-by-n list of integers, using A[0:1][ ] as the key
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A parallel mergesort of an array of 2-by-n integers.  Each key
// consists of two 32-bit or 64-bit unsigned integers.

#include "sort/GB_sort.h"

//------------------------------------------------------------------------------
// GB_msort_2_32_32
//------------------------------------------------------------------------------

#define GB_A0_t uint32_t
#define GB_A1_t uint32_t
#define GB_msort_2_binary_search      GB_msort_2_binary_search_32_32
#define GB_msort_2_create_merge_tasks GB_msort_2_create_merge_tasks_32_32
#define GB_msort_2_merge              GB_msort_2_merge_32_32
#define GB_msort_2_method             GB_msort_2_32_32

#include "sort/factory/GB_msort_2_template.c"

//------------------------------------------------------------------------------
// GB_msort_2_32_64
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_A1_t
#undef  GB_msort_2_binary_search
#undef  GB_msort_2_create_merge_tasks
#undef  GB_msort_2_merge
#undef  GB_msort_2_method

#define GB_A0_t uint32_t
#define GB_A1_t uint64_t
#define GB_msort_2_binary_search      GB_msort_2_binary_search_32_64
#define GB_msort_2_create_merge_tasks GB_msort_2_create_merge_tasks_32_64
#define GB_msort_2_merge              GB_msort_2_merge_32_64
#define GB_msort_2_method             GB_msort_2_32_64

#include "sort/factory/GB_msort_2_template.c"

//------------------------------------------------------------------------------
// GB_msort_2_64_32
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_A1_t
#undef  GB_msort_2_binary_search
#undef  GB_msort_2_create_merge_tasks
#undef  GB_msort_2_merge
#undef  GB_msort_2_method

#define GB_A0_t uint64_t
#define GB_A1_t uint32_t
#define GB_msort_2_binary_search      GB_msort_2_binary_search_64_32
#define GB_msort_2_create_merge_tasks GB_msort_2_create_merge_tasks_64_32
#define GB_msort_2_merge              GB_msort_2_merge_64_32
#define GB_msort_2_method             GB_msort_2_64_32

#include "sort/factory/GB_msort_2_template.c"

//------------------------------------------------------------------------------
// GB_msort_2_64_64
//------------------------------------------------------------------------------

#undef  GB_A0_t
#undef  GB_A1_t
#undef  GB_msort_2_binary_search
#undef  GB_msort_2_create_merge_tasks
#undef  GB_msort_2_merge
#undef  GB_msort_2_method

#define GB_A0_t uint64_t
#define GB_A1_t uint64_t
#define GB_msort_2_binary_search      GB_msort_2_binary_search_64_64
#define GB_msort_2_create_merge_tasks GB_msort_2_create_merge_tasks_64_64
#define GB_msort_2_merge              GB_msort_2_merge_64_64
#define GB_msort_2_method             GB_msort_2_64_64

#include "sort/factory/GB_msort_2_template.c"

//------------------------------------------------------------------------------
// GB_msort_2
//------------------------------------------------------------------------------

GrB_Info GB_msort_2     // sort array A of size 2-by-n
(
    void *restrict A_0,         // size n array
    bool A0_is_32,              // if true: A_0 is uint32, else uint64
    void *restrict A_1,         // size n array
    bool A1_is_32,              // if true: A_1 is uint32, else uint64
    const int64_t n,
    int nthreads_max            // max # of threads to use
)
{

    //--------------------------------------------------------------------------
    // handle small problems with a single thread
    //--------------------------------------------------------------------------

    int nthreads = GB_nthreads (n, GB_MSORT_BASECASE, nthreads_max) ;
    if (nthreads <= 1 || n <= GB_MSORT_BASECASE)
    { 
        // sequential quicksort
        GB_qsort_2 (A_0, A0_is_32, A_1, A1_is_32, n) ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // call the type-specific GB_msort_2 method
    //--------------------------------------------------------------------------

    if (A0_is_32)
    { 
        if (A1_is_32)
        { 
            return (GB_msort_2_32_32 (A_0, A_1, n, nthreads)) ;
        }
        else
        { 
            return (GB_msort_2_32_64 (A_0, A_1, n, nthreads)) ;
        }
    }
    else
    {
        if (A1_is_32)
        { 
            return (GB_msort_2_64_32 (A_0, A_1, n, nthreads)) ;
        }
        else
        { 
            return (GB_msort_2_64_64 (A_0, A_1, n, nthreads)) ;
        }
    }
}

