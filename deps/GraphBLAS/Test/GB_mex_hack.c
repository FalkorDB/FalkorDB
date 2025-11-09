//------------------------------------------------------------------------------
// GB_mex_hack: get or set the global hack flags
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "hack = GB_mex_hack (hack)"

#define NHACK 8

// current hacks (0-based):
//
//  0: saxpy3 balance
//  1: disable Werk
//  2: GPU control
//  3: disable the JIT entirely (returns GrB_NOT_IMPLEMENTED)
//  4: enable 32-bit methods

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
        int n = mxGetNumberOfElements (pargin [0]) ;
        hack = mxGetDoubles (pargin [0]) ;
        for (int k = 0 ; k < GB_IMIN (NHACK, n) ; k++)
        {
            GB_Global_hack_set (k, (int64_t) hack [k]) ;
        }
    }

    // GB_mex_hack returns an array of size NHACK
    pargout [0] = mxCreateDoubleMatrix (1, NHACK, mxREAL) ;
    hack = mxGetDoubles (pargout [0]) ;
    for (int k = 0 ; k < NHACK ; k++)
    {
        hack [k] = (double) GB_Global_hack_get (k) ;
    }
}

