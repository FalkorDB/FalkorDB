//------------------------------------------------------------------------------
// GB_mex_qsort_2: sort using GB_qsort_2
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "[I,J] = qsort (I,J)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    // check inputs
    if (nargin != 2 || nargout != 2)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }
    if (!mxIsClass (pargin [0], "int64"))
    {
        mexErrMsgTxt ("I must be a int64 array") ;
    }
    if (!mxIsClass (pargin [1], "int64"))
    {
        mexErrMsgTxt ("J must be a int64 array") ;
    }

    int64_t *I = mxGetData (pargin [0]) ;
    int64_t n = (uint64_t) mxGetNumberOfElements (pargin [0]) ;

    int64_t *J = mxGetData (pargin [1]) ;
    if (n != (uint64_t) mxGetNumberOfElements (pargin [1])) 
    {
        mexErrMsgTxt ("I and J must be the same length") ;
    }

    pargout [0] = GB_mx_create_full (n, 1, GrB_INT64) ;
    int64_t *Iout = mxGetData (pargout [0]) ;
    memcpy (Iout, I, n * sizeof (int64_t)) ;

    pargout [1] = GB_mx_create_full (n, 1, GrB_INT64) ;
    int64_t *Jout = mxGetData (pargout [1]) ;
    memcpy (Jout, J, n * sizeof (int64_t)) ;

    GB_qsort_2 (Iout, Jout, n) ;
}

