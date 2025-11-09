//------------------------------------------------------------------------------
// GB_cumsum1: cumlative sum of an array (single threaded)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Compute the cumulative sum of an array count[0:n], of size n+1:

//      count = cumsum ([0 count[0:n-1]]) ;

// That is, count [j] on input is overwritten with sum (count [0..j-1]).
// On input, count [n] is not accessed and is implicitly zero on input.
// On output, count [n] is the total sum.

#ifndef GB_CUMSUM1_H
#define GB_CUMSUM1_H

//------------------------------------------------------------------------------
// GB_cumsum1_64: uint64_t variant
//------------------------------------------------------------------------------

static inline bool GB_cumsum1_64    // cumulative sum of an array
(
    uint64_t *restrict count,       // size n+1, input/output
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

    uint64_t s = 0 ;
    for (int64_t i = 0 ; i < n ; i++)
    { 
        uint64_t c = count [i] ;
        count [i] = s ;
        s += c ;
    }
    count [n] = s ;

    return (true) ;     // do not check for integer overflow
}

//------------------------------------------------------------------------------
// GB_cumsum1_32: uint32_t variant
//------------------------------------------------------------------------------

// Returns true if successful, false if integer overflow occurs.

static inline bool GB_cumsum1_32   // cumulative sum of an array
(
    uint32_t *restrict count,   // size n+1, input/output
    const int64_t n
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (count != NULL) ;
    ASSERT (n >= 0) ;

    //--------------------------------------------------------------------------
    // check for overflow
    //--------------------------------------------------------------------------

    uint64_t s = 0 ;
    for (int64_t i = 0 ; i < n ; i++)
    {
        s += count [i] ;
        if (s > UINT32_MAX)
        { 
            return (false) ;
        }
    }

    //--------------------------------------------------------------------------
    // count = cumsum ([0 count[0:n-1]]) ;
    //--------------------------------------------------------------------------

    s = 0 ;
    for (int64_t i = 0 ; i < n ; i++)
    { 
        uint64_t c = count [i] ;
        count [i] = s ;
        s += c ;
    }
    count [n] = s ;

    return (true) ;
}

//------------------------------------------------------------------------------
// GB_cumsum1_float: float variant
//------------------------------------------------------------------------------

static inline bool GB_cumsum1_float   // cumulative sum of an array
(
    float *restrict count,   // size n+1, input/output
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

    double s = 0 ;
    for (int64_t i = 0 ; i < n ; i++)
    { 
        double c = count [i] ;
        count [i] = s ;
        s += c ;
    }
    count [n] = s ;

    return (true) ;
}

#endif

