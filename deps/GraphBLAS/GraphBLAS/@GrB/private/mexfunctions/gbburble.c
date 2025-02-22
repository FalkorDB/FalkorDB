//------------------------------------------------------------------------------
// gbburble: get/set the burble setting for diagnostic output
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage

// burble = gbburble ;
// burble = gbburble (burble) ;

#include "gb_interface.h"

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
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin <= 1 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // set the burble, if requested
    //--------------------------------------------------------------------------

    int32_t burble = false ;

    if (nargin > 0)
    { 
        // set the burble
        if (gb_mxarray_is_scalar (pargin [0]))
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
            ERROR ("input must be a scalar") ;
        }
        OK (GrB_Global_set_INT32 (GrB_GLOBAL, burble, GxB_BURBLE)) ;
    }

    //--------------------------------------------------------------------------
    // return the burble
    //--------------------------------------------------------------------------

    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &burble, GxB_BURBLE)) ;
    pargout [0] = mxCreateDoubleScalar (burble) ;
    gb_wrapup ( ) ;
}

