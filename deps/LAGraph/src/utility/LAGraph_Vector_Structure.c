//------------------------------------------------------------------------------
// LAGraph_Vector_Structure: return the structure of a vector
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

// LAGraph_Vector_Structure: return the structure of a vector as a boolean
// vector, where w(i)=true if the entry u(i) is present in the vector u.

#define LG_FREE_ALL GrB_free (w) ;

#include "LG_internal.h"

int LAGraph_Vector_Structure
(
    // output:
    GrB_Vector *w,  // a boolean matrix with same structure of u, with w(i)
                    // set to true if u(i) appears in the sparsity structure
                    // of u.
    // input:
    GrB_Vector u,
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    GrB_Index n ;
    LG_ASSERT_MSG (w != NULL, GrB_NULL_POINTER, "&w != NULL") ;
    LG_ASSERT (u != NULL, GrB_NULL_POINTER) ;
    (*w) = NULL ;

    //--------------------------------------------------------------------------
    // get the size of u
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Vector_size (&n, u)) ;

    //--------------------------------------------------------------------------
    // w<s(u)> = true
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Vector_new (w, GrB_BOOL, n)) ;
    GRB_TRY (GrB_assign (*w, u, NULL, (bool) true, GrB_ALL, n, GrB_DESC_S)) ;

    return (GrB_SUCCESS) ;
}
