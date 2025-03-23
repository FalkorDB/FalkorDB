//------------------------------------------------------------------------------
// GB_cumsum_float: cumlative sum of a float array
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Compute the cumulative sum of an array count[0:n], of size n+1:

//      count = cumsum ([0 count[0:n-1]]) ;

// That is, count [j] on input is overwritten with sum (count [0..j-1]).
// On input, count [n] is not accessed and is implicitly zero on input.
// On output, count [n] is the total sum.

#include "GB.h"

bool GB_cumsum_float            // cumulative sum of an array
(
    float *restrict count,      // size n+1, input/output
    const int64_t n,
    int nthreads,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (count != NULL) ;
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

    #define GB_CUMSUM1_TYPE GB_cumsum1_float
    #define GB_CHECK_OVERFLOW 0
    #define GB_NO_KRESULT
    #define GB_WS_TYPE double
    #include "cumsum/factory/GB_cumsum_template.c"

    return (true) ;
}

