//------------------------------------------------------------------------------
// GB_all_aliased: determine if two matrices are entirely aliased
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Returns true if A and B are the same or have the same content.  True if A ==
// B (or both NULL), or if all components A and B are aliased to each other.
// In the latter case, that component of A and B will always be shallow, in
// either A or B, or both.  NULL pointers are considered aliased.  The A->Y and
// B->Y hyperhashes are ignored.

#include "GB.h"

// true if pointers p1 and p2 are aliased, or both NULL
#define GB_POINTER_ALIASED(p1,p2) ((p1) == (p2))

bool GB_all_aliased         // determine if A and B are all aliased
(
    GrB_Matrix A,           // input A matrix
    GrB_Matrix B            // input B matrix
)
{

    //--------------------------------------------------------------------------
    // check the matrices themselves
    //--------------------------------------------------------------------------

    if (A == B)
    { 
        // two NULL matrices are equivalent
        return (true) ;
    }

    if (A == NULL || B == NULL)
    { 
        // one of A and B are non-NULL but one of them is NULL, so they are
        // not equal
        return (false) ;
    }

    //--------------------------------------------------------------------------
    // check their content
    //--------------------------------------------------------------------------

    bool all_aliased = 
        GB_POINTER_ALIASED (A->h, B->h) &&
        GB_POINTER_ALIASED (A->p, B->p) &&
        GB_POINTER_ALIASED (A->b, B->b) &&
        GB_POINTER_ALIASED (A->i, B->i) &&
        GB_POINTER_ALIASED (A->x, B->x) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (all_aliased) ;
}

