//------------------------------------------------------------------------------
// gbmex_nvals: number of entries in a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input may be either a GraphBLAS matrix struct or a standard built-in
// sparse matrix.

// Usage

// nvals = gbmex_nvals (A)

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_ALL
#define FREE_ALL GrB_Matrix_free (&A_to_free) ;

#define USAGE "usage: [nvals, nzmax] = gbmex_nvals (A)"

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

    GrB_Matrix A = NULL, A_to_free = NULL ;
    int arena = GrB_DEFAULT ;   // use default arena for temporary workspace

    GBMX_USAGE (nargin == 1 && nargout <= 2, USAGE) ;

    pargout [0] = mxCreateDoubleScalar (0) ;
    double *anvals_output = (double *) mxGetData (pargout [0]) ;
    double *anzmax_output = NULL, anzmax = 0 ;
    if (nargout > 1)
    { 
        pargout [1] = mxCreateDoubleScalar (0) ;
        anzmax_output = (double *) mxGetData (pargout [1]) ;
    }

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the input matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;

    //--------------------------------------------------------------------------
    // get the # of entries in the matrix
    //--------------------------------------------------------------------------

    uint64_t nvals ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;

    double anvals ;
    if (nvals == INT64_MAX)
    { 
        // A is a huge iso full matrix with too many entries to count with
        // a 64-bit integer.  anvals is recomputed in double, but it will
        // suffer roundoff errors.
        uint64_t nrows, ncols ;
        OK (GrB_Matrix_nrows (&nrows, A)) ;
        OK (GrB_Matrix_ncols (&ncols, A)) ;
        anvals = ((double) nrows) * ((double) ncols) ;
    }
    else
    { 
        anvals = (double) nvals ;
    }

    // get the # of entries that A can hold.  This ignores the iso property.
    if (nargout > 1)
    { 
        anzmax = (double) GB_helper11 (A) ;
        anzmax = fmax (anzmax, 1) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_ALL ;
    (*anvals_output) = anvals ;
    if (nargout > 1)
    { 
        (*anzmax_output) = anzmax ;
    }
    gb_wrapup ( ) ;
}

