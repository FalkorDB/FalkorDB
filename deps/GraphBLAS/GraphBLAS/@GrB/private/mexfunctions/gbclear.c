//------------------------------------------------------------------------------
// gbclear: set all global GraphBLAS settings to their defaults for MATLAB
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method does not clear the JIT hash table.  To do that, use:
//
//      GrB.jit ('off') ; GrB.jit ('on') ;

#include "gb_interface.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // get test coverage
    //--------------------------------------------------------------------------

    #ifdef GBCOV
    gbcov_get ( ) ;
    #endif

    //--------------------------------------------------------------------------
    // initialize GraphBLAS if necessary
    //--------------------------------------------------------------------------

    gb_usage (true, "") ;

    //--------------------------------------------------------------------------
    // set global defaults
    //--------------------------------------------------------------------------

    gb_defaults ( ) ;

    //--------------------------------------------------------------------------
    // save test coverage
    //--------------------------------------------------------------------------

    GB_WRAPUP ;
}

