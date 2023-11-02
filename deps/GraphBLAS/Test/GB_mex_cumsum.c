//------------------------------------------------------------------------------
// GB_mex_cumsum: cumulative using GB_cumsum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "[p,k] = GB_mex_cumsum (c,nthreads,malloc_debug_count)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    bool malloc_debug = GB_mx_get_global (true) ;

    // check inputs
    if (nargin < 1 || nargin > 3 || nargout > 2)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    if (!mxIsClass (pargin [0], "int64"))
    {
        mexErrMsgTxt ("c must be an int64 array") ;
    }

    int64_t *c = mxGetData (pargin [0]) ;
    int64_t n = (uint64_t) mxGetNumberOfElements (pargin [0]) ;

    int GET_SCALAR (1, int, nthreads, 1) ;

    int GET_SCALAR (2, int, malloc_debug_count, 2) ;

    // make a copy of the input array (as a row vector)
    pargout [0] = GB_mx_create_full (1, n+1, GrB_INT64) ;
    int64_t *p = mxGetData (pargout [0]) ;
    memcpy (p, c, n * sizeof (int64_t)) ;
    p [n] = 0 ;

    // create the 2nd output, kresult, if requested
    int64_t *kresult = NULL ;
    if (nargout > 1)
    {
        pargout [1] = GB_mx_create_full (1, 1, GrB_INT64) ;
        kresult = mxGetData (pargout [1]) ;
    }

    if (!malloc_debug)
    {
        // normal usage
        GB_cumsum (p, n, kresult, nthreads, NULL) ;
    }
    else
    {
        // test with malloc failures
        // printf ("test cumsum with malloc_debug_count: %d\n", malloc_debug_count) ;
        GB_Global_malloc_debug_set (true) ;
        GB_Global_malloc_debug_count_set (malloc_debug_count) ;
        GB_cumsum (p, n, kresult, nthreads, NULL) ;
        GB_Global_malloc_debug_set (false) ;
    }

    // log the test coverage
    GB_mx_put_global (true) ;   
}

