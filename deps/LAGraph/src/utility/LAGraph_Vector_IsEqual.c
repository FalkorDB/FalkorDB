//------------------------------------------------------------------------------
// LAGraph_Vector_IsEqual: check two vectors for exact equality
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

// LAGraph_Vector_IsEqual: checks if two vectors are identically equal (same
// size, type, structure, size, and values).

// See also LAGraph_Matrix_IsEqual.

// If the two vectors are GrB_FP32, GrB_FP64, or related, and have NaNs, then
// this function will return false, since NaN == NaN is false.  To check for
// NaN equality (like isequalwithequalnans in MATLAB), use
// LAGraph_Vector_IsEqualOp with a user-defined operator f(x,y) that returns
// true if x and y are both NaN.

#define LG_FREE_WORK GrB_free (&C) ;

#include "LG_internal.h"

int LAGraph_Vector_IsEqual
(
    // output:
    bool *result,           // true if A == B, false if A != B or error
    // input:
    const GrB_Vector A,
    const GrB_Vector B,
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    GrB_Vector C = NULL ;
    LG_ASSERT (result != NULL, GrB_NULL_POINTER) ;

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
    // compare the type of A and B
    //--------------------------------------------------------------------------

    char atype_name [LAGRAPH_MAX_NAME_LEN] ;
    char btype_name [LAGRAPH_MAX_NAME_LEN] ;
    LG_TRY (LAGraph_Vector_TypeName (atype_name, A, msg)) ;
    LG_TRY (LAGraph_Vector_TypeName (btype_name, B, msg)) ;
    if (!MATCHNAME (atype_name, btype_name))
    {
        // types differ
        (*result) = false ;
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
    // get the GrB_EQ_type operator
    //--------------------------------------------------------------------------

    GrB_Type type ;
    LG_TRY (LAGraph_TypeFromName (&type, atype_name, msg)) ;
    GrB_BinaryOp op = NULL ;
    // select the comparator operator
    if      (type == GrB_BOOL  ) op = GrB_EQ_BOOL   ;
    else if (type == GrB_INT8  ) op = GrB_EQ_INT8   ;
    else if (type == GrB_INT16 ) op = GrB_EQ_INT16  ;
    else if (type == GrB_INT32 ) op = GrB_EQ_INT32  ;
    else if (type == GrB_INT64 ) op = GrB_EQ_INT64  ;
    else if (type == GrB_UINT8 ) op = GrB_EQ_UINT8  ;
    else if (type == GrB_UINT16) op = GrB_EQ_UINT16 ;
    else if (type == GrB_UINT32) op = GrB_EQ_UINT32 ;
    else if (type == GrB_UINT64) op = GrB_EQ_UINT64 ;
    else if (type == GrB_FP32  ) op = GrB_EQ_FP32   ;
    else if (type == GrB_FP64  ) op = GrB_EQ_FP64   ;
    #if 0
    else if (type == GxB_FC32  ) op = GxB_EQ_FC32   ;
    else if (type == GxB_FC64  ) op = GxB_EQ_FC64   ;
    #endif

    LG_ASSERT_MSG (op != NULL, GrB_NOT_IMPLEMENTED, "type not supported") ;

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
