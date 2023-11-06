//------------------------------------------------------------------------------
// GB_mex_hack: get or set the global hack flags
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "hack = GB_mex_hack (hack)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    double *hack ;

    if (nargin > 1 || nargout > 1)
    {
        mexErrMsgTxt ("usage: " USAGE "\n") ;
    }

    if (nargin == 1)
    {
        if (mxGetNumberOfElements (pargin [0]) != 2)
        {
            mexErrMsgTxt ("usage: " USAGE " where length(hack) is 2\n") ;
        }
        hack = mxGetDoubles (pargin [0]) ;
        GB_Global_hack_set (0, (int64_t) hack [0]) ;
        GB_Global_hack_set (1, (int64_t) hack [1]) ;
    }

    // GB_mex_hack returns an array of size 2
    pargout [0] = mxCreateDoubleMatrix (1, 2, mxREAL) ;
    hack = mxGetDoubles (pargout [0]) ;
    hack [0] = (double) GB_Global_hack_get (0) ;
    hack [1] = (double) GB_Global_hack_get (1) ;
}

