//------------------------------------------------------------------------------
// GB_mex_diag: compute C=diag(A,k)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = diag (A,k), where A and C are double
// C is a matrix the same size as A, not a vector.

#include "GB_mex.h"

#define USAGE "C = GB_mex_diag (A,k)"

#define FREE_ALL                        \
{                                       \
    GrB_Scalar_free_(&Thunk) ;          \
    GrB_Matrix_free_(&A) ;              \
    GrB_Matrix_free_(&C) ;              \
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

    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL, C = NULL ;
    GrB_Scalar Thunk = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 1 || nargin > 2)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // get A
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("failed") ;
    }

    int64_t k = 0 ;
    // get k
    if (nargin > 1)
    {
        k = (int64_t) mxGetScalar (pargin [1]) ;
    }

    // construct C
    METHOD (GrB_Matrix_new (&C, GrB_FP64, A->vlen, A->vdim)) ;

    #undef GET_DEEP_COPY
    #undef FREE_DEEP_COPY

    #define GET_DEEP_COPY  GrB_Matrix_new (&C, GrB_FP64, A->vlen, A->vdim) ;
    #define FREE_DEEP_COPY GrB_Matrix_free_(&C) ;

    GrB_Scalar_new (&Thunk, GrB_INT64) ;
    GrB_Scalar_setElement_INT64_(Thunk, k) ;
    GrB_Scalar_wait_(Thunk, GrB_MATERIALIZE) ;

    // C = diag (A,k)
    METHOD (GxB_Matrix_select_(C, NULL, NULL, GxB_DIAG, A, Thunk, NULL)) ;

    // return C as a regular built-in sparse matrix
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C diag", false) ;

    FREE_ALL ;
}

