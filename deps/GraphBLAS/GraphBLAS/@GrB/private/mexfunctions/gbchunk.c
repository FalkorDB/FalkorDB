//------------------------------------------------------------------------------
// gbchunk: get/set the chunk size to use in GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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

    GrB_Scalar chunk = NULL ;

    if (nargin > 0)
    { 
        // set the chunk
        CHECK_ERROR (!gb_mxarray_is_scalar (pargin [0]),
            "input must be a scalar") ;
        chunk = (GrB_Scalar) gb_get_shallow (pargin [0]) ;
        OK (GrB_Global_set_Scalar (GrB_GLOBAL, chunk, GxB_CHUNK)) ;
        OK (GrB_Scalar_free (&chunk)) ;
    }

    //--------------------------------------------------------------------------
    // get the chunk and return it
    //--------------------------------------------------------------------------

    OK (GrB_Scalar_new (&chunk, GrB_FP64)) ;
    OK (GrB_Global_get_Scalar (GrB_GLOBAL, chunk, GxB_CHUNK)) ;

    //--------------------------------------------------------------------------
    // return the chunk
    //--------------------------------------------------------------------------

    pargout [0] = gb_export ((GrB_Matrix *) &chunk, KIND_FULL) ;
    gb_wrapup ( ) ;
}

