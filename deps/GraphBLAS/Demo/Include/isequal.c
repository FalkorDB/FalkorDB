//------------------------------------------------------------------------------
// isequal: check two matrices for exact equality
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// isequal: check if two matrices are identically equal (same size,type,
// pattern, size, and values).  Checking for the same type requires a function
// that is an extension in SuiteSparse:GraphBLAS.  For the standard API, there
// is no way to determine the type of a matrix.

// isequal_type: uses just the standard API.

// For both methods, if the two matrices are FP32 or FP64, and have NaNs, then
// these functions will return false, since NaN == NaN is false.  To check for
// NaN equality, use isequal_type with a user-defined operator f(x,y) that
// returns true if x and y are both NaN.

#include "GraphBLAS.h"
#include "graphblas_demos.h"

// call a GraphBLAS method and return if an error occurs
#undef  OK
#define OK(method)                                          \
{                                                           \
    GrB_Info info = method ;                                \
    if (! (info == GrB_SUCCESS || info == GrB_NO_VALUE))    \
    {                                                       \
        /* error occured: free workspace and return */      \
        GrB_Matrix_free (&C) ;                              \
        return (info) ;                                     \
    }                                                       \
}

//------------------------------------------------------------------------------
// isequal_type: check two matrices, works in any GraphBLAS
//------------------------------------------------------------------------------

GrB_Info isequal_type       // return GrB_SUCCESS if successful
(
    bool *result,           // true if A == B, false if A != B or error
    GrB_Matrix A,
    GrB_Matrix B,
    GrB_BinaryOp op         // should be GrB_EQ_<type>, for the type of A and B
)
{

    GrB_Matrix C = NULL ;
    GrB_Index nrows1, ncols1, nrows2, ncols2, nvals, nvals1, nvals2 ;

    if (result == NULL)
    {
        // error: required parameter, result, is NULL
        return (GrB_NULL_POINTER) ;
    }
    (*result) = false ;

    // check the size of A and B
    OK (GrB_Matrix_nrows (&nrows1, A)) ;
    OK (GrB_Matrix_nrows (&nrows2, B)) ;
    if (nrows1 != nrows2)
    {
        // # of rows differ
        return (GrB_SUCCESS) ;    
    }

    OK (GrB_Matrix_ncols (&ncols1, A)) ;
    OK (GrB_Matrix_ncols (&ncols2, B)) ;
    if (ncols1 != ncols2)
    {
        // # of cols differ
        return (GrB_SUCCESS) ;
    }

    // check the # entries in A and B
    OK (GrB_Matrix_nvals (&nvals1, A)) ;
    OK (GrB_Matrix_nvals (&nvals2, B)) ;
    if (nvals1 != nvals2)
    {
        // # of entries differ
        return (GrB_SUCCESS) ;
    }

    // C = A .* B, where the pattern of C is the intersection of A and B
    OK (GrB_Matrix_new (&C, GrB_BOOL, nrows1, ncols1)) ;
    OK (GrB_Matrix_eWiseMult_BinaryOp (C, NULL, NULL, op, A, B, NULL)) ;

    // ensure C has the same number of entries as A and B
    OK (GrB_Matrix_nvals (&nvals, C)) ;
    if (nvals != nvals1)
    {
        // pattern of A and B are different
        GrB_Matrix_free (&C) ;
        return (GrB_SUCCESS) ;
    }

    // result = and (C)
    OK (GrB_Matrix_reduce_BOOL (result, NULL, GrB_LAND_MONOID_BOOL, C, NULL)) ;

    // free workspace and return result
    GrB_Matrix_free (&C) ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// isequal: for SuiteSparse/GraphBLAS only; also check if types are the same
//------------------------------------------------------------------------------

#ifdef GxB_SUITESPARSE_GRAPHBLAS
// the isequal function only works with SuiteSparse:GraphBLAS

GrB_Info isequal            // return GrB_SUCCESS if successful
(
    bool *result,           // true if A == B, false if A != B or error
    GrB_Matrix A,
    GrB_Matrix B,
    GrB_BinaryOp userop     // for A and B with user-defined types.  ignored
                            // if A and B are of built-in types
)
{
    GrB_Matrix C = NULL ;
    GrB_Type atype, btype ;
    GrB_BinaryOp op ;

    if (result == NULL)
    {
        // error: required parameter, result, is NULL
        return (GrB_NULL_POINTER) ;
    }
    (*result) = false ;

    // check the type of A and B
    OK (GxB_Matrix_type (&atype, A)) ;
    OK (GxB_Matrix_type (&btype, B)) ;
    if (atype != btype)
    {
        // types differ
        return (GrB_SUCCESS) ;
    }

    // select the comparator operator
    if      (atype == GrB_BOOL  ) op = GrB_EQ_BOOL ;
    else if (atype == GrB_INT8  ) op = GrB_EQ_INT8 ;
    else if (atype == GrB_INT16 ) op = GrB_EQ_INT16 ;
    else if (atype == GrB_INT32 ) op = GrB_EQ_INT32 ;
    else if (atype == GrB_INT64 ) op = GrB_EQ_INT64 ;
    else if (atype == GrB_UINT8 ) op = GrB_EQ_UINT8 ;
    else if (atype == GrB_UINT16) op = GrB_EQ_UINT16 ;
    else if (atype == GrB_UINT32) op = GrB_EQ_UINT32 ;
    else if (atype == GrB_UINT64) op = GrB_EQ_UINT64 ;
    else if (atype == GrB_FP32  ) op = GrB_EQ_FP32   ;
    else if (atype == GrB_FP64  ) op = GrB_EQ_FP64   ;
    else if (atype == GxB_FC32  ) op = GxB_EQ_FC32   ;
    else if (atype == GxB_FC64  ) op = GxB_EQ_FC64   ;
    else                          op = userop ; // A and B are user-defined

    // check the size, pattern, and values of A and B
    OK (isequal_type (result, A, B, op)) ;

    // return result
    return (GrB_SUCCESS) ;
}
#endif

