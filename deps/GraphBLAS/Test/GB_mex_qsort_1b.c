//------------------------------------------------------------------------------
// GB_mex_qsort_1b: sort using GB_qsort_1b_generic
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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
    if (!mxIsClass (pargin [0], "uint64"))
    {
        mexErrMsgTxt ("I must be a uint64 array") ;
    }
    if (!mxIsClass (pargin [1], "uint64"))
    {
        mexErrMsgTxt ("J must be a uint64 array") ;
    }

    uint64_t *I = mxGetData (pargin [0]) ;
    int64_t n = (uint64_t) mxGetNumberOfElements (pargin [0]) ;

    uint64_t *J = mxGetData (pargin [1]) ;
    if (n != (uint64_t) mxGetNumberOfElements (pargin [1])) 
    {
        mexErrMsgTxt ("I and J must be the same length") ;
    }

    pargout [0] = GB_mx_create_full (n, 1, GrB_UINT64) ;
    uint64_t *Iout = mxGetData (pargout [0]) ;
    memcpy (Iout, I, n * sizeof (uint64_t)) ;

    pargout [1] = GB_mx_create_full (n, 1, GrB_UINT64) ;
    uint64_t *Jout = mxGetData (pargout [1]) ;
    memcpy (Jout, J, n * sizeof (uint64_t)) ;

//  double t = GB_omp_get_wtime ( ) ;
    GB_qsort_1b_64_generic (Iout, (GB_void *) Jout, sizeof (int64_t), n) ;
//  printf ("1b_64_generic time %g\n", GB_omp_get_wtime ( ) - t) ;
}

