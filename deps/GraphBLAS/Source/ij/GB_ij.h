//------------------------------------------------------------------------------
// GB_ij.h: definitions for I and J index lists
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_IJ_H
#define GB_IJ_H

#include "GB.h"

void GB_ijlength            // get the length and kind of an index list I
(
    const void *I,          // list of indices (actual or implicit)
    const bool I_is_32,     // if true, I is 32-bit; else 64-bit
    const int64_t ni,       // length I, or special
    const int64_t limit,    // indices must be in the range 0 to limit-1
    int64_t *nI,            // actual length of I
    int *Ikind,             // kind of I: GB_ALL, GB_RANGE, GB_STRIDE, GB_LIST
    int64_t Icolon [3]      // begin:inc:end for all but GB_LIST
) ;

GrB_Info GB_ijproperties        // check I and determine its properties
(
    // input:
    const void *I,              // list of indices, or special
    const bool I_is_32,         // if true, I is 32-bit; else 64-bit
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
    // input:
    const void *I,              // size ni, where ni > 1 always holds
    const bool I_is_32,
    const int64_t ni,           // length I
    const int64_t imax,         // maximum value in I 
    // output:
    int64_t *p_ni2,             // # of indices in I2
    void **p_I2,                // size ni2, where I2 [0..ni2-1] contains the
                                // sorted indices with duplicates removed.
    bool *I2_is_32_handle,      // if I2_is_32 true, I2 is 32 bits; else 64 bits
    size_t *I2_size_handle,
    void **p_I2k,               // output array of size ni2
    bool *I2k_is_32_handle,     // if I2k_is_32 true, I2 is 32 bits; else 64
    size_t *I2k_size_handle,
    GB_Werk Werk
) ;

GrB_Info GB_ijxvector
(
    // input:
    GrB_Vector List,        // defines the list of integers, either from
                            // List->x or List-i.  If List is NULL, it defines
                            // I = GrB_ALL.
    bool need_copy,         // if true, I must be allocated
    int which,              // 0: I list, 1: J list, 2: X list
    const GrB_Descriptor desc,  // row_list, col_list, val_list descriptors
    bool is_build,          // if true, method is GrB_build; otherwise, it is
                            // assign, subassign, or extract
    // output:
    void **I_handle,        // the list I; may be GrB_ALL
    int64_t *ni_handle,     // the length of I, or special (GxB_RANGE)
    size_t *I_size_handle,  // if > 0, I has been allocated by this
                            // method.  Otherwise, it is a shallow pointer into
                            // List->x or List->i.
    GrB_Type *I_type_handle,    // the type of I: GrB_UINT32 or GrB_UINT64 for
                            // assign, subassign, extract, or for build with
                            // the descriptor uses the indices.  For build,
                            // this is List->type when using the values.
    GB_Werk Werk                            
) ;

//------------------------------------------------------------------------------
// GB_ij_is_in_list: determine if i is in list I
//------------------------------------------------------------------------------

// Given i and I, return true if there is a k so that i is the kth item in I.
// The value of k is not returned.

static inline bool GB_ij_is_in_list // determine if i is in the list I
(
    const void *I,              // list of indices for GB_LIST
    const bool I_is_32,         // if true, I is 32-bit; else 64-bit
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
        uint64_t ui = (uint64_t) i ;
        found = GB_binary_search (ui, I, I_is_32, &pleft, &pright) ;
        return (found) ;
    }
}

#endif

