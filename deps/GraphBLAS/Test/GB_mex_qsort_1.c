//------------------------------------------------------------------------------
// GB_mex_qsort_1: sort using GB_qsort_1
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "[I] = qsort (I)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    // check inputs
    if (nargin != 1 || nargout != 1)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    bool I_is_32 ;
    if (mxIsClass (pargin [0], "uint32"))
    { 
        I_is_32 = true ;
    }
    else if (mxIsClass (pargin [0], "uint64"))
    {
        I_is_32 = false ;
    }
    else
    {
        mexErrMsgTxt ("I must be a uint32 or uint64 array") ;
    }

    void *I = mxGetData (pargin [0]) ;
    int64_t n = (uint64_t) mxGetNumberOfElements (pargin [0]) ;

    pargout [0] = GB_mx_create_full (n, 1, I_is_32 ? GrB_UINT32 : GrB_UINT64) ;
    void *Iout = mxGetData (pargout [0]) ;
    memcpy (Iout, I, n * (I_is_32 ? sizeof (uint32_t) : sizeof (uint64_t))) ;

//  double t = GB_omp_get_wtime ( ) ;
    GB_qsort_1 (Iout, I_is_32, n) ;
//  printf ("qsort_1 time %g\n", GB_omp_get_wtime ( ) - t) ;
}

