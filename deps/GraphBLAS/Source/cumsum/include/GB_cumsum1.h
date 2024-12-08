//------------------------------------------------------------------------------
// GB_cumsum1: cumlative sum of an array (single threaded)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Compute the cumulative sum of an array count[0:n], of size n+1:

//      count = cumsum ([0 count[0:n-1]]) ;

// That is, count [j] on input is overwritten with sum (count [0..j-1]).
// On input, count [n] is not accessed and is implicitly zero on input.
// On output, count [n] is the total sum.

#ifndef GB_CUMSUM1_H
#define GB_CUMSUM1_H

static inline void GB_cumsum1    // cumulative sum of an array
(
    int64_t *restrict count,     // size n+1, input/output
    const int64_t n
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (count != NULL) ;
    ASSERT (n >= 0) ;

    //--------------------------------------------------------------------------
    // count = cumsum ([0 count[0:n-1]]) ;
    //--------------------------------------------------------------------------

    int64_t s = 0 ;
    for (int64_t i = 0 ; i < n ; i++)
    { 
        int64_t c = count [i] ;
        count [i] = s ;
        s += c ;
    }
    count [n] = s ;
}

#endif

