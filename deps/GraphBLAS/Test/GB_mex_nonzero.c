//------------------------------------------------------------------------------
// GB_mex_nonzero: compute C=nonzero(A)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = nonzero (A), where A and C are double

#include "GB_mex.h"

#define USAGE "C = GB_mex_nonzero (A)"

#define FREE_ALL                        \
{                                       \
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

    // check inputs
    if (nargout > 1 || nargin != 1)
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

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // construct C
    METHOD (GrB_Matrix_new (&C, GrB_FP64, A->vlen, A->vdim)) ;

    #undef GET_DEEP_COPY
    #undef FREE_DEEP_COPY

    #define GET_DEEP_COPY  GrB_Matrix_new (&C, GrB_FP64, A->vlen, A->vdim) ;
    #define FREE_DEEP_COPY GrB_Matrix_free_(&C) ;

    // C = nonzero (A)
    METHOD (GxB_Matrix_select_(C, NULL, NULL, GxB_NONZERO, A, NULL, NULL)) ;

    // return C as a regular built-in sparse matrix
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C nonzero", false) ;

    FREE_ALL ;
}

