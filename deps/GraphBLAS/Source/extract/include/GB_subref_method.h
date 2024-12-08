//------------------------------------------------------------------------------
// GB_subref_method.h: definitions of GB_subref_method and GB_subref_work
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_SUBREF_METHOD_H
#define GB_SUBREF_METHOD_H

//------------------------------------------------------------------------------
// GB_subref_method: select a method for C(:,k) = A(I,j), for one vector of C
//------------------------------------------------------------------------------

// Determines the method used for to construct C(:,k) = A(I,j) for a
// single vector of C and A.

static inline int GB_subref_method  // return the method to use (1 to 12)
(
    // input:
    const int64_t ajnz,             // nnz (A (:,j))
    const int64_t avlen,            // A->vlen
    const int Ikind,                // GB_ALL, GB_RANGE, GB_STRIDE, or GB_LIST
    const int64_t nI,               // length of I
    const bool I_inverse_ok,        // true if I is invertable
    const bool need_qsort,          // true if C(:,k) requires sorting
    const int64_t iinc,             // increment for GB_STRIDE
    const bool I_has_duplicates     // true if duplicates in I
                                    // (false if not yet known)
)
{

    //--------------------------------------------------------------------------
    // determine the method to use for C(:,k) = A (I,j)
    //--------------------------------------------------------------------------

    int method ;

    if (ajnz == avlen)
    {
        // A(:,j) is dense
        if (Ikind == GB_ALL)
        { 
            // Case 1: C(:,k) = A(:,j) are both dense
            method = 1 ;
        }
        else
        { 
            // Case 2: C(:,k) = A(I,j), where A(:,j) is dense,
            // for Ikind == GB_RANGE, GB_STRIDE, or GB_LIST
            method = 2 ;
        }
    }
    else if (nI == 1)
    { 
        // Case 3: one index
        method = 3 ;
    }
    else if (Ikind == GB_ALL)
    { 
        // Case 4: I is ":"
        method = 4 ;
    }
    else if (Ikind == GB_RANGE)
    { 
        // Case 5: C (:,k) = A (ibegin:iend,j)
        method = 5 ;
    }
    else if ((Ikind == GB_LIST && !I_inverse_ok) ||  // must do Case 6
        (64 * nI < ajnz))    // Case 6 faster
    { 
        // Case 6: nI not large; binary search of A(:,j) for each i in I
        method = 6 ;
    }
    else if (Ikind == GB_STRIDE)
    {
        if (iinc >= 0)
        { 
            // Case 7: I = ibegin:iinc:iend with iinc >= 0
            method = 7 ;
        }
        else if (iinc < -1)
        { 
            // Case 8: I = ibegin:iinc:iend with iinc < =1
            method = 8 ;
        }
        else // iinc == -1
        { 
            // Case 9: I = ibegin:(-1):iend
            method = 9 ;
        }
    }
    else // Ikind == GB_LIST, and I inverse buckets will be used
    {
        // construct the I inverse buckets
        if (need_qsort)
        { 
            // Case 10: nI large, need qsort
            // duplicates are possible so cjnz > ajnz can hold.  If fine tasks
            // use this method, a post sort is needed when all tasks are done.
            method = 10 ;
        }
        else if (I_has_duplicates)
        { 
            // Case 11: nI large, no qsort, with duplicates
            // duplicates are possible so cjnz > ajnz can hold.  Note that the
            // # of duplicates is only known after I is inverted, which might
            // not yet be done.  In that case, nuplicates is assumed to be
            // zero, and Case 12 is assumed to be used instead.  This is
            // revised after I is inverted.
            method = 11 ;
        }
        else
        { 
            // Case 12: nI large, no qsort, no duplicates
            method = 12 ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (method) ;
}

//------------------------------------------------------------------------------
// GB_subref_work: return the work for each subref method
//------------------------------------------------------------------------------

static inline int64_t GB_subref_work   // return the work for a subref method
(
    // output
    bool *p_this_needs_I_inverse,   // true if I needs to be inverted
    // input:
    const int64_t ajnz,             // nnz (A (:,j))
    const int64_t avlen,            // A->vlen
    const int Ikind,                // GB_ALL, GB_RANGE, GB_STRIDE, or GB_LIST
    const int64_t nI,               // length of I
    const bool I_inverse_ok,        // true if I is invertable
    const bool need_qsort,          // true if C(:,k) requires sorting
    const int64_t iinc              // increment for GB_STRIDE
)
{

    //--------------------------------------------------------------------------
    // get the method
    //--------------------------------------------------------------------------

    // nduplicates in I not yet known; it is found when I is inverted.  For
    // now, assume I has no duplicate entries.  All that is needed for now is
    // the work required for each C(:,k), and whether or not I inverse must be
    // created.  The # of duplicates has no impact on the I inverse decision,
    // and a minor effect on the work (which is ignored).  Method 11 is only
    // used if I_has_duplicates is true.

    const bool I_has_duplicates = false ;   // not yet known

    int method = GB_subref_method (ajnz, avlen, Ikind, nI, I_inverse_ok,
        need_qsort, iinc, I_has_duplicates) ;

    //--------------------------------------------------------------------------
    // get the work
    //--------------------------------------------------------------------------

    int64_t work = 0 ;
    switch (method)
    {
        case  1 : work = nI ;           break ;
        case  2 : work = nI ;           break ;
        case  3 : work = 1 ;            break ;
        case  4 : work = ajnz ;         break ;
        case  5 : work = ajnz ;         break ;
        case  6 : work = nI * 64 ;      break ;
        case  7 : work = ajnz ;         break ;
        case  8 : work = ajnz ;         break ;
        case  9 : work = ajnz ;         break ;
        case 10 : work = ajnz * 32 ;    break ;
//      case 11 :
//                work = ajnz * 2 ;     break ; // case not determined yet
        default :
        case 12 : work = ajnz ;         break ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*p_this_needs_I_inverse) = (method >= 10) ;
    return (work) ;
}

#endif

