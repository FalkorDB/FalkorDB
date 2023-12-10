//------------------------------------------------------------------------------
// GB_mex_dump: copy and print a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "C = GB_mex_dump (A,pr)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&A) ;              \
    Complex_finalize ( ) ;              \
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

    bool malloc_debug = GB_mx_get_global (false) ;
    GrB_Matrix A = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 1 || nargin > 2)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get A (deep copy)
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A input", true, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get pr
    int GET_SCALAR (1, int, pr, 1) ;

    // dump the matrix
    GrB_Info info = GB_Matrix_check (A, "", pr, NULL) ;
    if (info != GrB_SUCCESS)
    {
        mexErrMsgTxt ("matrix fail") ;
    }

    // return A as a struct and free the GraphBLAS A
    pargout [0] = GB_mx_Matrix_to_mxArray (&A, "C output", true) ;

    FREE_ALL ;
}

