//------------------------------------------------------------------------------
// gbisequal: isequal (A,B)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbisequal returns isequal(A,B) for two matrices A and B.

// Usage:

//  result = gbisequal (A,B)

#include "gb_interface.h"

#define USAGE "usage: s = GrB.isequal (A, B)"

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

    gb_usage (nargin == 2 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get the arguments
    //--------------------------------------------------------------------------

    GrB_Matrix A = gb_get_shallow (pargin [0]) ;
    GrB_Matrix B = gb_get_shallow (pargin [1]) ;

    //--------------------------------------------------------------------------
    // check if they are equal
    //--------------------------------------------------------------------------

    pargout [0] = mxCreateLogicalScalar (gb_is_equal (A, B)) ;

    //--------------------------------------------------------------------------
    // free shallow copies
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Matrix_free (&B)) ;
    GB_WRAPUP ;
}

