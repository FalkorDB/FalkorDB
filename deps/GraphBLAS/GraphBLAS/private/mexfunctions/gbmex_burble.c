//------------------------------------------------------------------------------
// gbmex_burble: get/set the burble setting for diagnostic output
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage

// burble = gbmex_burble ;
// burble = gbmex_burble (burble) ;

#include "gb_interface.h"
#include "gbmx_interface.h"

#define USAGE "usage: burble = GrB.burble ; or GrB.burble (burble)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs and construct outputs
    //--------------------------------------------------------------------------

    GBMX_USAGE (nargin <= 1 && nargout <= 1, USAGE) ;

    pargout [0] = mxCreateDoubleScalar (0) ;
    double *burble_output = (double *) mxGetData (pargout [0]) ;

    //--------------------------------------------------------------------------
    // get input burble
    //--------------------------------------------------------------------------

    int32_t burble = false ;
    if (nargin > 0)
    { 
        if (gbmx_mxarray_is_scalar (pargin [0]))
        { 
            // argument is a numeric scalar
            burble = (int32_t) mxGetScalar (pargin [0]) ;
        }
        else if (mxIsLogicalScalar (pargin [0]))
        { 
            // argument is a logical scalar
            burble = (int32_t) mxIsLogicalScalarTrue (pargin [0]) ;
        }
        else
        { 
            ERROR ("input must be a scalar", GrB_INVALID_VALUE) ;
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // set the burble, if requested
    //--------------------------------------------------------------------------

    if (nargin > 0)
    { 
        OK (GrB_Global_set_INT32 (GrB_GLOBAL, burble, GxB_BURBLE)) ;
    }

    //--------------------------------------------------------------------------
    // return the burble
    //--------------------------------------------------------------------------

    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &burble, GxB_BURBLE)) ;
    (*burble_output) = (double) burble ;
    gb_wrapup ( ) ;
}

