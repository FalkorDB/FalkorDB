//------------------------------------------------------------------------------
// gbchunk: get/set the chunk size to use in GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// chunk = gbchunk ;
// chunk = gbchunk (chunk) ;

#include "gb_interface.h"

#define USAGE "usage: c = GrB.chunk ; or GrB.chunk (c)"

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
    // set the chunk, if requested
    //--------------------------------------------------------------------------

    double c ;

    if (nargin > 0)
    { 
        // set the chunk
        CHECK_ERROR (!gb_mxarray_is_scalar (pargin [0]),
            "input must be a scalar") ;
        c = (double) mxGetScalar (pargin [0]) ;
        OK (GxB_Global_Option_set (GxB_CHUNK, c)) ;
    }

    //--------------------------------------------------------------------------
    // return the chunk
    //--------------------------------------------------------------------------

    OK (GxB_Global_Option_get (GxB_CHUNK, &c)) ;
    pargout [0] = mxCreateDoubleScalar (c) ;
    GB_WRAPUP ;
}

