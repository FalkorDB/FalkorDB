//------------------------------------------------------------------------------
// GB_mex_AplusB_M_aliased: compute C<B>=A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This is for testing only.  See GrB_eWiseAdd instead.  Returns a plain
// built-in matrix, in double.

#include "GB_mex.h"

#define USAGE "C = GB_mex_AplusB_M_aliased (A, B, op)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&A) ;              \
    GrB_Matrix_free_(&B) ;              \
    GrB_Matrix_free_(&C) ;              \
    GrB_free (&op) ;                    \
    GB_mx_put_global (true) ;           \
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    struct GB_Matrix_opaque C_header ;
    GrB_Matrix C = NULL ;

    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL ;
    GrB_Matrix B = NULL ;
    GrB_BinaryOp op = NULL ;

    GB_WERK (USAGE) ;

    // check inputs
    if (nargout > 1 || nargin != 3)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get A and B
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A", false, true) ;
    B = GB_mx_mxArray_to_Matrix (pargin [1], "B", false, true) ;
    if (A == NULL || B == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("failed") ;
    }

    // get op
    bool user_complex = (Complex != GxB_FC64)
        && (A->type == Complex || B->type == Complex) ;
    if (!GB_mx_mxArray_to_BinaryOp (&op, pargin [2], "op",
        A->type, user_complex) || op == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("op failed") ;
    }

    // C<B> = A+B using the op.  M == B alias

// this can ignore the mask now; so use GB_eWiseAdd instead:
//  bool ignore ;
//  METHOD (GB_add (C, A->type, true, B, false, false, &ignore, A, B,
//      false, NULL, NULL, op, false, Werk)) ;

    // using eWiseAdd instead:
    GrB_Index nrows, ncols ;
    GrB_Matrix_nrows (&nrows, A) ;
    GrB_Matrix_ncols (&ncols, A) ;

    #define GET_DEEP_COPY                                   \
    GrB_Matrix_new (&C, op->ztype, nrows, ncols) ;          \
    GxB_Matrix_Option_set (C, GxB_FORMAT, GxB_BY_COL) ;

    #define FREE_DEEP_COPY                                  \
    GrB_Matrix_free_(&C) ;

    GET_DEEP_COPY ;
    METHOD (GrB_Matrix_eWiseAdd_BinaryOp (C, B, NULL, op, A, B, NULL)) ;

    // return C as a plain sparse matrix
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C<B>=A+B result", false) ;

    FREE_ALL ;
}

