//------------------------------------------------------------------------------
// GB_ijlist.h: return kth item, i = I [k], in an index list
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_IJLIST_H
#define GB_IJLIST_H

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
        // ASSERT (Ikind == GB_LIST) ;
        // ASSERT (I != NULL) ;
        return (I [k]) ;
    }
}

#endif

