//------------------------------------------------------------------------------
// gbthreads: get/set the maximum # of threads to use in GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// nthreads = gbthreads
// nthreads = gbthreads (nthreads)

#include "gb_interface.h"

#define USAGE "usage: nthreads = GrB.threads ; or GrB.threads (nthreads)"

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
    // set the # of threads, if requested
    //--------------------------------------------------------------------------

    int nthreads_max ;

    if (nargin > 0)
    { 
        // set the # of threads
        CHECK_ERROR (!gb_mxarray_is_scalar (pargin [0]),
            "input must be a scalar") ;
        nthreads_max = (int) mxGetScalar (pargin [0]) ;
        OK (GxB_Global_Option_set (GxB_NTHREADS, nthreads_max)) ;
    }

    //--------------------------------------------------------------------------
    // return # of threads
    //--------------------------------------------------------------------------

    OK (GxB_Global_Option_get (GxB_NTHREADS, &nthreads_max)) ;
    pargout [0] = mxCreateDoubleScalar (nthreads_max) ;
    GB_WRAPUP ;
}

