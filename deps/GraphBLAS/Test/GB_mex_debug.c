//------------------------------------------------------------------------------
// GB_mex_debug: determine GB_DEBUG status
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "[debug compact malloc cover] = GB_mex_debug"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    bool malloc_debug = GB_mx_get_global (false) ;

    // check inputs
    if (nargout > 4 || nargin != 0)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    bool pr = (nargout == 0) ;

    if (pr)
    {
        printf ("\n-------------------------------------------------------\n") ;
        printf ("GraphBLAS compilation and run-time options:\n") ;
    }

    #ifdef GB_DEBUG
    if (pr) printf ("GB_DEBUG:     debugging enabled:"
                                 " GraphBLAS will be slow\n") ;
    pargout [0] = mxCreateDoubleScalar (1) ;
    #else
    if (pr) printf ("GB_DEBUG:     normal: debugging not enabled\n") ;
    pargout [0] = mxCreateDoubleScalar (0) ;
    #endif

    pargout [1] = mxCreateDoubleScalar (0) ;

    if (malloc_debug)
    {
        if (pr) printf ("malloc debug: enabled: malloc testing\n") ;
        pargout [2] = mxCreateDoubleScalar (1) ;
    }
    else
    {
        if (pr) printf ("malloc debug: normal: no malloc testing\n") ;
        pargout [2] = mxCreateDoubleScalar (0) ;
    }

    #ifdef GBCOVER
    if (pr) printf ("GBCOVER:      enabled: test coverage\n") ;
    pargout [3] = mxCreateDoubleScalar (1) ;
    #else
    if (pr) printf ("GBCOVER:      normal: no statement coverage\n") ;
    pargout [3] = mxCreateDoubleScalar (0) ;
    #endif

    if (pr)
    {
        printf ("-------------------------------------------------------\n\n") ;
    }

    GB_mx_put_global (false) ;
}

