//------------------------------------------------------------------------------
// gbmex_chunk: get/set the chunk size to use in GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// chunk = gbmex_chunk ;
// chunk = gbmex_chunk (chunk) ;

// GrB* and mx* methods are intermingled, since the GrB methods do not allocate
// any memory.

#include "gb_interface.h"
#include "gbmx_interface.h"

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

    GBMX_USAGE (nargin <= 1 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // set the chunk, if requested
    //--------------------------------------------------------------------------

    double chunk ;
    if (nargin > 0)
    { 
        CHECK_ERROR (!gbmx_mxarray_is_scalar (pargin [0]),
            "input must be a scalar") ;
        chunk = mxGetScalar (pargin [0]) ;
        OK (GxB_Global_Option_set_FP64 (GxB_CHUNK, chunk)) ;
    }

    //--------------------------------------------------------------------------
    // return the chunk
    //--------------------------------------------------------------------------

    OK (GxB_Global_Option_get_FP64 (GxB_CHUNK, &chunk)) ;
    pargout [0] = mxCreateDoubleScalar (chunk) ;
    gb_wrapup ( ) ;
}

