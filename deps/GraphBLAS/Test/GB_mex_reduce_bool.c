//------------------------------------------------------------------------------
// GB_mex_reduce_bool: c = accum(c,reduce_to_scalar(A)) for boolean
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reduce a boolean matrix or vector to a boolean.

#include "GB_mex.h"

#define USAGE "result = GB_mex_reduce_bool (A, op, identity, terminal)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&A) ;              \
    GrB_Monoid_free_(&reduce) ;         \
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
    GrB_Info info ;
    GrB_Matrix A = NULL ;
    GrB_Monoid reduce = NULL ;
    GrB_BinaryOp reduceop = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 3 || nargin > 4)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

/*
    if (A->type != GrB_BOOL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A must be boolean") ;
    }
*/

    // get the op (always boolean)
    if (!GB_mx_mxArray_to_BinaryOp (&reduceop, pargin [1], "reduceop",
        GrB_BOOL, false) || reduceop == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("reduceop failed") ;
    }

    // get the boolean identity value
    bool GET_SCALAR (2, bool, identity, true) ;

    // get the boolean terminal value, if any.  default is true
    bool terminal = true ;
    bool has_terminal = (nargin == 4) ;
    if (has_terminal)
    {
        GET_SCALAR (3, bool, terminal, true) ;
    }

    // create the reduce monoid
    if (has_terminal)
    {
        info = GxB_Monoid_terminal_new_BOOL_(&reduce, reduceop, identity, terminal) ;
    }
    else
    {
        info = GrB_Monoid_new_BOOL_(&reduce, reduceop, identity) ;
    }

    if (info != GrB_SUCCESS)
    {
        FREE_ALL ;
        mexErrMsgTxt ("monoid failed") ;
    }

    if (GB_Global_burble_get ( )) GxB_print (reduce, 3) ;

    // reduce to a scalar
    bool result = false ;
    info = GrB_Matrix_reduce_BOOL_(&result, NULL, reduce, A, NULL) ;

    if (info != GrB_SUCCESS)
    {
        FREE_ALL ;
        mexErrMsgTxt ("GrB_reduce failed") ;
    }

    // return result as a boolean scalar
    pargout [0] = GB_mx_create_full (1, 1, GrB_BOOL) ;
    GB_void *p = mxGetData (pargout [0]) ;
    memcpy (p, &result, sizeof (bool)) ;

    FREE_ALL ;
}

