//------------------------------------------------------------------------------
// gbmex_threads: get/set the maximum # of threads to use in GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// nthreads = gbmex_threads
// nthreads = gbmex_threads (nthreads)

// GrB* and mx* methods are intermingled, since the GrB methods do not allocate
// any memory.

#include "gb_interface.h"
#include "gbmx_interface.h"

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

    GBMX_USAGE (nargin <= 1 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // set the # of threads, if requested
    //--------------------------------------------------------------------------

    int nthreads ;
    if (nargin > 0)
    { 
        CHECK_ERROR (!gbmx_mxarray_is_scalar (pargin [0]),
            "input must be a scalar") ;
        nthreads = (int) mxGetScalar (pargin [0]) ;
        OK (GrB_Global_set_INT32 (GrB_GLOBAL, nthreads, GxB_NTHREADS)) ;
    }

    //--------------------------------------------------------------------------
    // return # of threads
    //--------------------------------------------------------------------------

    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &nthreads, GxB_NTHREADS)) ;
    pargout [0] = mxCreateDoubleScalar ((double) nthreads) ;
    gb_wrapup ( ) ;
}

