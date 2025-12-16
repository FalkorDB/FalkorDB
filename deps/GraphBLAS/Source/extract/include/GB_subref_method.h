//------------------------------------------------------------------------------
// GB_subref_method.h: definitions of GB_subref_method and GB_subref_work
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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
    const bool need_qsort,          // true if C(:,k) requires sorting
    const int64_t iinc              // increment for GB_STRIDE
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
    else if (64 * nI < ajnz)    // Case 6 faster in this case
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
    else // Ikind == GB_LIST, and R = inverse(I) will be used
    {
        // construct the R matrix
        if (need_qsort)
        { 
            // Case 10: nI large, need qsort
            // duplicates are possible so cjnz > ajnz can hold.  If fine tasks
            // use this method, a post sort is needed when all tasks are done.
            method = 10 ;
        }
        else
        {
            // Case 11: nI large, no qsort, duplicates are OK
            method = 11 ;
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
    const bool need_qsort,          // true if C(:,k) requires sorting
    const int64_t iinc              // increment for GB_STRIDE
)
{

    //--------------------------------------------------------------------------
    // get the method
    //--------------------------------------------------------------------------

    int method = GB_subref_method (ajnz, avlen, Ikind, nI, need_qsort, iinc) ;

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
        default :
        case 11 : work = ajnz * 2 ;     break ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*p_this_needs_I_inverse) = (method >= 10) ;
    return (work) ;
}

#endif

