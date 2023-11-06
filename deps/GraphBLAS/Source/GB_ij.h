//------------------------------------------------------------------------------
// GB_ij.h: definitions for I and J index lists
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_IJ_H
#define GB_IJ_H

#include "GB.h"

void GB_ijlength            // get the length and kind of an index list I
(
    const GrB_Index *I,     // list of indices (actual or implicit)
    const int64_t ni,       // length I, or special
    const int64_t limit,    // indices must be in the range 0 to limit-1
    int64_t *nI,            // actual length of I
    int *Ikind,             // kind of I: GB_ALL, GB_RANGE, GB_STRIDE, GB_LIST
    int64_t Icolon [3]      // begin:inc:end for all but GB_LIST
) ;

GrB_Info GB_ijproperties        // check I and determine its properties
(
    // input:
    const GrB_Index *I,         // list of indices, or special
    const int64_t ni,           // length I, or special
    const int64_t nI,           // actual length from GB_ijlength
    const int64_t limit,        // I must be in the range 0 to limit-1
    // input/output:
    int *Ikind,                 // kind of I, from GB_ijlength
    int64_t Icolon [3],         // begin:inc:end from GB_ijlength
    // output:
    bool *I_is_unsorted,        // true if I is out of order
    bool *I_has_dupl,           // true if I has a duplicate entry (undefined
                                // if I is unsorted)
    bool *I_is_contig,          // true if I is a contiguous list, imin:imax
    int64_t *imin_result,       // min (I)
    int64_t *imax_result,       // max (I)
    GB_Werk Werk
) ;

GrB_Info GB_ijsort
(
    const GrB_Index *restrict I, // size ni, where ni > 1 always holds
    int64_t *restrict p_ni,      // : size of I, output: # of indices in I2
    GrB_Index *restrict *p_I2,   // size ni2, where I2 [0..ni2-1]
                        // contains the sorted indices with duplicates removed.
    size_t *I2_size_handle,
    GrB_Index *restrict *p_I2k,  // output array of size ni2
    size_t *I2k_size_handle
) ;

// given k, return the kth item i = I [k] in the list
static inline int64_t GB_ijlist     // get the kth item in a list of indices
(
    const GrB_Index *I,         // list of indices
    const int64_t k,            // return i = I [k], the kth item in the list
    const int Ikind,            // GB_ALL, GB_RANGE, GB_STRIDE, or GB_LIST
    const int64_t Icolon [3]    // begin:inc:end for all but GB_LIST
)
{
    if (Ikind == GB_ALL)
    { 
        // I is ":"
        return (k) ;
    }
    else if (Ikind == GB_RANGE)
    { 
        // I is begin:end
        return (Icolon [GxB_BEGIN] + k) ;
    }
    else if (Ikind == GB_STRIDE)
    { 
        // I is begin:inc:end
        // note that iinc can be negative or even zero
        return (Icolon [GxB_BEGIN] + k * Icolon [GxB_INC]) ;
    }
    else // Ikind == GB_LIST
    { 
        ASSERT (Ikind == GB_LIST) ;
        ASSERT (I != NULL) ;
        return (I [k]) ;
    }
}

// given i and I, return true there is a k so that i is the kth item in I
static inline bool GB_ij_is_in_list // determine if i is in the list I
(
    const GrB_Index *I,         // list of indices for GB_LIST
    const int64_t nI,           // length of I if Ikind is GB_LIST
    int64_t i,                  // find i = I [k] in the list
    const int Ikind,            // GB_ALL, GB_RANGE, GB_STRIDE, or GB_LIST
    const int64_t Icolon [3]    // begin:inc:end for all but GB_LIST
)
{

    if (Ikind == GB_ALL)
    { 
        // I is ":", all indices are in the sequence
        return (true) ;
    }
    else if (Ikind == GB_RANGE)
    { 
        // I is begin:end
        int64_t b = Icolon [GxB_BEGIN] ;
        int64_t e = Icolon [GxB_END] ;
        if (i < b) return (false) ;
        if (i > e) return (false) ;
        return (true) ;
    }
    else if (Ikind == GB_STRIDE)
    {
        // I is begin:inc:end
        // note that inc can be negative or even zero
        int64_t b   = Icolon [GxB_BEGIN] ;
        int64_t inc = Icolon [GxB_INC] ;
        int64_t e   = Icolon [GxB_END] ;
        if (inc == 0)
        { 
            // lo:stride:hi with stride of zero.
            // I is empty if inc is zero, so i is not in I.
            return (false) ;
        }
        else if (inc > 0)
        { 
            // forward direction, increment is positive
            // I = b:inc:e = [b, b+inc, b+2*inc, ..., e]
            if (i < b) return (false) ;
            if (i > e) return (false) ;
            // now i is in the range [b ... e]
            ASSERT (b <= i && i <= e) ;
            i = i - b ;
            ASSERT (0 <= i && i <= (e-b)) ;
            // the sequence I-b = [0, inc, 2*inc, ... e-b].
            // i is in the sequence if i % inc == 0
            return (i % inc == 0) ;
        }
        else // inc < 0
        { 
            // backwards direction, increment is negative
            inc = -inc ;
            // now inc is positive
            ASSERT (inc > 0) ;
            // I = b:(-inc):e = [b, b-inc, b-2*inc, ... e]
            if (i > b) return (false) ;
            if (i < e) return (false) ;
            // now i is in the range of the sequence, [b down to e]
            ASSERT (e <= i && i <= b) ;
            i = b - i ;
            ASSERT (0 <= i && i <= (b-e)) ;
            // b-I = 0:(inc):(b-e) = [0, inc, 2*inc, ... (b-e)]
            // i is in the sequence if i % inc == 0
            return (i % inc == 0) ;
        }
    }
    else // Ikind == GB_LIST
    { 
        ASSERT (Ikind == GB_LIST) ;
        ASSERT (I != NULL) ;
        // search for i in the sorted list I
        bool found ;
        int64_t pleft = 0 ;
        int64_t pright = nI-1 ;
        if (i < 0) return (false) ;
        GrB_Index ui = (GrB_Index) i ;
        GB_BINARY_SEARCH (ui, I, pleft, pright, found) ;
        return (found) ;
    }
}

#endif

