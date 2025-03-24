//------------------------------------------------------------------------------
// GB_cumsum: cumlative sum of an integer array (uint32_t or uint64_t)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Compute the cumulative sum of an array count[0:n], of size n+1:

//      k = sum (count [0:n-1] != 0) ;
//      count = cumsum ([0 count[0:n-1]]) ;

// That is, count [j] on input is overwritten with sum (count [0..j-1]).
// On input, count [n] is not accessed and is implicitly zero on input.
// On output, count [n] is the total sum.

// If count is uint32, returns true if OK, false if overflow.  The overflow
// condition is not checked if count is uint64_t (always returns true).

#include "GB.h"

bool GB_cumsum                  // cumulative sum of an array
(
    void *restrict count_arg,   // size n+1, input/output
    bool count_is_32,           // if true: count is uint32_t, else uint64_t
    const int64_t n,
    int64_t *restrict kresult,  // return k, if needed by the caller
    int nthreads,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (count_arg != NULL) ;
    ASSERT (n >= 0) ;

    //--------------------------------------------------------------------------
    // determine # of threads to use
    //--------------------------------------------------------------------------

    #if !defined ( _OPENMP )
    nthreads = 1 ;
    #endif

    if (nthreads > 1)
    { 
        nthreads = GB_IMIN (nthreads, n / GB_CHUNK_DEFAULT) ;
        nthreads = GB_IMAX (nthreads, 1) ;
    }

    //--------------------------------------------------------------------------
    // count = cumsum ([0 count[0:n-1]]) ;
    //--------------------------------------------------------------------------

    #define GB_WS_TYPE uint64_t

    if (count_is_32)
    { 
        uint32_t *restrict count = (uint32_t *) count_arg ;
        #define GB_CUMSUM1_TYPE GB_cumsum1_32
        #define GB_CHECK_OVERFLOW 1
        #include "cumsum/factory/GB_cumsum_template.c"
    }
    else
    { 
        uint64_t *restrict count = (uint64_t *) count_arg ;
        #define GB_CUMSUM1_TYPE GB_cumsum1_64
        #define GB_CHECK_OVERFLOW 0
        #include "cumsum/factory/GB_cumsum_template.c"
    }

    return (true) ;
}

