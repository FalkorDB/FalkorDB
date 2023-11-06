//------------------------------------------------------------------------------
// gbburble: get/set the burble setting for diagnostic output
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage

// b = gbburble ;
// b = gbburble (b) ;

#include "gb_interface.h"

#define USAGE "usage: b = GrB.burble ; or GrB.burble (b)"

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

    bool b ;

    if (nargin > 0)
    { 
        // set the burble
        if (gb_mxarray_is_scalar (pargin [0]))
        { 
            // argument is a numeric scalar
            b = (bool) mxGetScalar (pargin [0]) ;
        }
        else if (mxIsLogicalScalar (pargin [0]))
        { 
            // argument is a logical scalar
            b = (bool) mxIsLogicalScalarTrue (pargin [0]) ;
        }
        else
        { 
            ERROR ("input must be a scalar") ;
        }
        OK (GxB_Global_Option_set (GxB_BURBLE, b)) ;
    }

    //--------------------------------------------------------------------------
    // return the burble
    //--------------------------------------------------------------------------

    OK (GxB_Global_Option_get (GxB_BURBLE, &b)) ;
    pargout [0] = mxCreateDoubleScalar (b) ;
    GB_WRAPUP ;
}

