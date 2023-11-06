//------------------------------------------------------------------------------
// GB_mex_binaryop: parse a binaryop, for testing
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "GB_mex_binaryop (binaryop_struct))"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    // check inputs
    if (nargin != 1)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    GrB_BinaryOp binaryop = NULL ;
    GB_mx_mxArray_to_BinaryOp (&binaryop, pargin [0], "binaryop",
        GrB_FP64, false) ;

    GrB_Info info = GB_BinaryOp_check (binaryop, "binaryop", GxB_COMPLETE,
        stdout) ;
    if (info != GrB_SUCCESS)
    {
        mexErrMsgTxt ("binaryop failed") ;
    }
}

