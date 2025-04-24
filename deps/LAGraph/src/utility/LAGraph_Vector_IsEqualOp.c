//------------------------------------------------------------------------------
// LAGraph_Vector_IsEqualOp: compare two vectors with a given op
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

// LAGraph_Vector_IsEqualOp: checks if two vectors are equal (same size,
// structure, size, and values), using a provided binary operator.

// See also LAGraph_Matrix_IsEqualOp.

#define LG_FREE_WORK GrB_free (&C) ;

#include "LG_internal.h"

//------------------------------------------------------------------------------
// LAGraph_Vector_IsEqualOp:  compare two vectors using a given operator
//------------------------------------------------------------------------------

int LAGraph_Vector_IsEqualOp
(
    // output:
    bool *result,           // true if A == B, false if A != B or error
    // input:
    const GrB_Vector A,
    const GrB_Vector B,
    const GrB_BinaryOp op,        // comparator to use
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    GrB_Vector C = NULL ;
    LG_ASSERT (op != NULL && result != NULL, GrB_NULL_POINTER) ;

    //--------------------------------------------------------------------------
    // check for NULL and aliased vectors
    //--------------------------------------------------------------------------

    if (A == NULL || B == NULL || A == B)
    {
        // two NULL vectors are identical, as are two aliased matrices
        (*result) = (A == B) ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // compare the size of A and B
    //--------------------------------------------------------------------------

    GrB_Index nrows1, nrows2;
    GRB_TRY (GrB_Vector_size (&nrows1, A)) ;
    GRB_TRY (GrB_Vector_size (&nrows2, B)) ;
    if (nrows1 != nrows2)
    {
        // # of rows differ
        (*result) = false ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // compare the # entries in A and B
    //--------------------------------------------------------------------------

    GrB_Index nvals1, nvals2 ;
    GRB_TRY (GrB_Vector_nvals (&nvals1, A)) ;
    GRB_TRY (GrB_Vector_nvals (&nvals2, B)) ;
    if (nvals1 != nvals2)
    {
        // # of entries differ
        (*result) = false ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // C = A .* B, where the structure of C is the intersection of A and B
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Vector_new (&C, GrB_BOOL, nrows1)) ;
    GRB_TRY (GrB_eWiseMult (C, NULL, NULL, op, A, B, NULL)) ;

    //--------------------------------------------------------------------------
    // ensure C has the same number of entries as A and B
    //--------------------------------------------------------------------------

    GrB_Index nvals ;
    GRB_TRY (GrB_Vector_nvals (&nvals, C)) ;
    if (nvals != nvals1)
    {
        // structure of A and B are different
        LG_FREE_WORK ;
        (*result) = false ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // result = and (C)
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_reduce (result, NULL, GrB_LAND_MONOID_BOOL, C, NULL)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    LG_FREE_WORK ;
    return (GrB_SUCCESS) ;
}
