//------------------------------------------------------------------------------
// gbthreads: get/set the maximum # of threads to use in GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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
        OK (GrB_Global_set_INT32 (GrB_GLOBAL, nthreads_max, GxB_NTHREADS)) ;

    }

    //--------------------------------------------------------------------------
    // return # of threads
    //--------------------------------------------------------------------------

    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &nthreads_max, GxB_NTHREADS)) ;
    pargout [0] = mxCreateDoubleScalar (nthreads_max) ;
    gb_wrapup ( ) ;
}

