//------------------------------------------------------------------------------
// gbtiming: get/set the diagnostic timings
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage

// t = gbtiming ;       // get the current timing
// gbtiming (0) ;       // clear the timing

#include "gb_interface.h"

#define USAGE "usage: t = GrB.timing (clear) ;"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin <= 1 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get the current timing
    //--------------------------------------------------------------------------

    pargout [0] = mxCreateDoubleMatrix (40, 1, mxREAL) ;
    double *T = mxGetData (pargout [0]) ;
    for (int k = 0 ; k < 40 ; k++)
    {
        T [k] = GB_Global_timing_get (k) ;
    }

    //--------------------------------------------------------------------------
    // clear the current timing, if requested
    //--------------------------------------------------------------------------

    if (nargin > 0 && mxGetScalar (pargin [0]) == 0)
    {
        GB_Global_timing_clear_all ( ) ;
    }
    GB_WRAPUP ;
}

