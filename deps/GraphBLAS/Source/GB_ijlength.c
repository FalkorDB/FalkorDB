//------------------------------------------------------------------------------
// GB_ijlength: get the length and kind of an index list I
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Determine the length of I, and process the colon notation I = begin:inc:end.
// No error checking is done.

#include "GB_ij.h"

// ensure an unsigned integer does not cause signed integer overflow
#define GB_LIMIT(u) (int64_t) (GB_IMIN (u, INT64_MAX))

void GB_ijlength            // get the length and kind of an index list I
(
    const GrB_Index *I,     // list of indices (actual or implicit)
    const int64_t ni,       // length I, or special
    const int64_t limit,    // indices must be in the range 0 to limit-1
    int64_t *nI,            // actual length of I
    int *Ikind,             // kind of I: GB_ALL, GB_RANGE, GB_STRIDE, GB_LIST
    int64_t Icolon [3]      // begin:inc:end for all but GB_LIST
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (I != NULL) ;
    ASSERT (limit >= 0) ;
    ASSERT (limit <= GB_NMAX) ;

    //--------------------------------------------------------------------------
    // determine the length of I
    //--------------------------------------------------------------------------

    if (I == GrB_ALL)
    { 

        //----------------------------------------------------------------------
        // I = ":" = 0:limit-1
        //----------------------------------------------------------------------

        (*Ikind) = GB_ALL ;

        Icolon [GxB_BEGIN] = 0 ;
        Icolon [GxB_INC  ] = 1 ;
        Icolon [GxB_END  ] = limit-1 ;

        (*nI) = limit ;

    }
    else if (ni == GxB_RANGE)
    {

        //----------------------------------------------------------------------
        // I = ibegin:iend
        //----------------------------------------------------------------------

        (*Ikind) = GB_RANGE ;

        // the array I must have size at least 2

        int64_t ibegin = GB_LIMIT (I [GxB_BEGIN]) ;
        int64_t iend   = GB_LIMIT (I [GxB_END  ]) ;

        ASSERT (ibegin >= 0) ;

        if (ibegin == 0 && iend == limit-1)
        { 
            // 0:limit-1 is the same as ":"
            (*Ikind) = GB_ALL ;
        }

        Icolon [GxB_BEGIN] = ibegin ;
        Icolon [GxB_INC  ] = 1 ;
        Icolon [GxB_END  ] = iend ;

        if (ibegin > iend)
        { 
            // the list is empty
            (*nI) = 0 ;
        }
        else // ibegin <= iend
        { 
            // list ibegin:iend is not empty
            (*nI) = (iend - ibegin + 1) ;
        }

    }
    else if (ni == GxB_STRIDE)
    {

        //----------------------------------------------------------------------
        // I = ibegin:iinc:iend where iinc >= 0
        //----------------------------------------------------------------------

        (*Ikind) = GB_STRIDE ;

        // The array I must have size at least 3.  It is an unsigned uint64_t
        // array, so integers must be positive.

        int64_t ibegin = GB_LIMIT (I [GxB_BEGIN]) ;
        int64_t iinc   = GB_LIMIT (I [GxB_INC  ]) ;
        int64_t iend   = GB_LIMIT (I [GxB_END  ]) ;

        ASSERT (ibegin >= 0) ;
        ASSERT (iinc   >= 0) ;
        ASSERT (iend   >= 0) ;

        if (iinc == 1)
        {
            if (ibegin == 0 && iend == limit-1)
            { 
                // 0:1:limit-1 is the same as ":"
                (*Ikind) = GB_ALL ;
            }
            else
            { 
                // ibegin:1:iend is the same as ibegin:iend
                (*Ikind) = GB_RANGE ;
            }
        }

        // an increment of 0 means the list is empty
        if (iinc == 0)
        { 
            (*nI) = 0 ;
        }
        else // iinc > 0
        {
            if (ibegin > iend)
            { 
                // the list ibegin:iinc:iend is empty (for example 10:1:0)
                (*nI) = 0 ;
            }
            else // ibegin <= iend
            { 
                // the list is non-empty (for example 4:2:7 = [4 6])
                (*nI) = ((iend - ibegin) / iinc) + 1 ;
            }
        }

        Icolon [GxB_BEGIN] = ibegin ;
        Icolon [GxB_INC  ] = iinc ;
        Icolon [GxB_END  ] = iend ;

    }
    else if (ni == GxB_BACKWARDS)
    {

        //----------------------------------------------------------------------
        // I = ibegin:iinc:iend where iinc <= 0
        //----------------------------------------------------------------------

        (*Ikind) = GB_STRIDE ;

        // The array I must have size at least 3.  It is an unsigned uint64_t
        // array, so integers must be positive.

        int64_t ibegin = GB_LIMIT (I [GxB_BEGIN]) ;
        int64_t iinc   = GB_LIMIT (I [GxB_INC  ]) ;
        int64_t iend   = GB_LIMIT (I [GxB_END  ]) ;

        ASSERT (iinc   >= 0) ;

        // the stride is backwards, so negate iinc
        iinc = -iinc ;

        ASSERT (ibegin >= 0) ;
        ASSERT (iinc   <= 0) ;
        ASSERT (iend   >= 0) ;

        // an increment of 0 means the list is empty
        if (iinc == 0)
        { 
            (*nI) = 0 ;
        }
        else // iinc < 0
        {
            if (ibegin < iend)
            { 
                // the list ibegin:iinc:iend is empty (for example 1:-1:10)
                (*nI) = 0 ;
            }
            else
            { 
                // the list is non-empty (for example 7:-2:4 = [7 5])
                // two positive numbers are divided here
                (*nI) = ((ibegin - iend) / (-iinc)) + 1 ;
            }
        }

        Icolon [GxB_BEGIN] = ibegin ;
        Icolon [GxB_INC  ] = iinc ;
        Icolon [GxB_END  ] = iend ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // I is an array of indices
        //----------------------------------------------------------------------

        (*Ikind) = GB_LIST ;

        // not computed
        Icolon [GxB_BEGIN] = 0 ;
        Icolon [GxB_INC  ] = 0 ;
        Icolon [GxB_END  ] = 0 ;

        (*nI) = ni ;
    }
}

