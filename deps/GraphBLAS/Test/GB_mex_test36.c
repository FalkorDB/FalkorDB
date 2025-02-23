//------------------------------------------------------------------------------
// GB_mex_test36: reduce a huge iso full matrix to a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

/*
    m = 2^40 ;
    n = 2^48 ;
    H = pi * GrB.ones (m, n) ;
    H
    s = sum (H, 'all') ;
*/

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info ;
    GrB_Matrix H = NULL ;

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    bool malloc_debug = GB_mx_get_global (true) ;
    int expected = GrB_SUCCESS ;

    //--------------------------------------------------------------------------
    // create and reduce a huge iso full matrix
    //--------------------------------------------------------------------------

    uint64_t nrows = 1UL << 40 ;
    uint64_t ncols = 1UL << 48 ;
    double pi = 3.141592653589793 ;
    double sum = 0 ;
    mexPrintf (
      "\nCreating huge iso-valued full matrix,\n"
        "size m=2^40 by n=2^48, with all entries equal to pi:\n") ;
    OK (GrB_Matrix_new (&H, GrB_FP64, nrows, ncols)) ;
    OK (GrB_Matrix_assign_FP64 (H, NULL, NULL, pi, GrB_ALL, nrows,
        GrB_ALL, ncols, NULL)) ;
    OK (GrB_Matrix_wait (H, GrB_MATERIALIZE)) ;
    OK (GxB_Matrix_fprint (H, "H", GxB_COMPLETE_VERBOSE, NULL)) ;
    OK (GrB_Matrix_reduce_FP64 (&sum, NULL, GrB_PLUS_MONOID_FP64, H, NULL)) ;
    OK (GrB_Matrix_free (&H)) ;
    double truth = pi * ((double) nrows) * ((double) ncols) ;
    mexPrintf ("\npi*m*n:             %g\n", truth) ;
    mexPrintf ("sum of all entries: %g\n", sum) ;
    mexPrintf ("absolute error:     %g\n", sum - truth) ;
    double relerr = fabs (sum - truth) / truth ;
    mexPrintf ("relative error:     %g\n", relerr) ;
    CHECK (relerr < 1e-14) ;
    pargout [0] = mxCreateDoubleScalar (sum) ;

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test36: all tests passed\n\n") ;
}

