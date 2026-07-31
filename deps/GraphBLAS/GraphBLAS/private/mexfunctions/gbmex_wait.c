//------------------------------------------------------------------------------
// gbmex_wait: finish work in a GhB handle matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbmex_wait (G)

// finishes all pending work in a GhB matrix.  Does nothing if the input is not
// a GhB handle matrix from GraphBLAS v10.4.0 or later.

#include "gb_interface.h"
#include "gbmx_interface.h"

#define USAGE "usage: gbmex_wait (G)"

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

    GBMX_USAGE (nargin == 1, USAGE) ;

    //--------------------------------------------------------------------------
    // wait on the matrix
    //--------------------------------------------------------------------------

    GrB_Matrix A = gbmx_get_ghb_matrix (pargin [0]) ;
    if (A != NULL)
    {
        OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
    }

    //--------------------------------------------------------------------------
    // return the result
    //--------------------------------------------------------------------------

    gb_wrapup ( ) ;
}

