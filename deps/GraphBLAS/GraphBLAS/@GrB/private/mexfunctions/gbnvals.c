//------------------------------------------------------------------------------
// gbnvals: number of entries in a GraphBLAS matrix struct
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input may be either a GraphBLAS matrix struct or a standard built-in
// sparse matrix.

// Usage

// nvals = gbnvals (X)

#include "gb_interface.h"

#define USAGE "usage: nvals = gbnvals (X)"

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

    gb_usage (nargin == 1 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get a shallow copy of the matrix
    //--------------------------------------------------------------------------

    GrB_Matrix X = gb_get_shallow (pargin [0]) ;

    //--------------------------------------------------------------------------
    // get the # of entries in the matrix
    //--------------------------------------------------------------------------

    GrB_Index nvals ;
    OK (GrB_Matrix_nvals (&nvals, X)) ;

    //--------------------------------------------------------------------------
    // free the shallow copy and return the result
    //--------------------------------------------------------------------------

    double anvals ;
    if (nvals == INT64_MAX)
    {
        GrB_Index nrows, ncols ;
        OK (GrB_Matrix_nrows (&nrows, X)) ;
        OK (GrB_Matrix_ncols (&ncols, X)) ;
        anvals = ((double) nrows) * ((double) ncols) ;
    }
    else
    {
        anvals = (double) nvals ;
    }
    pargout [0] = mxCreateDoubleScalar (anvals) ;
    OK (GrB_Matrix_free (&X)) ;
    GB_WRAPUP ;
}

