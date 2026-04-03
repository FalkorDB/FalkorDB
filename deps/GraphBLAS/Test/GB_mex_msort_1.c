//------------------------------------------------------------------------------
// GB_mex_msort_1: sort using GB_msort_1
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "[I] = GB_mex_msort_1 (I,nthreads)"

#define WALLCLOCK GB_omp_get_wtime ( )

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
    if (nargin != 2 || nargout != 1)
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

    int GET_SCALAR (1, int, nthreads, 1) ;

    // make a copy of the input arrays

    pargout [0] = GB_mx_create_full (n, 1, I_is_32 ? GrB_UINT32 : GrB_UINT64) ;
    void *Iout = mxGetData (pargout [0]) ;
    memcpy (Iout, I, n * (I_is_32 ? sizeof (uint32_t) : sizeof (uint64_t))) ;

    double t = WALLCLOCK ;
    GB_msort_1 (Iout, I_is_32, n, nthreads) ;
    t = WALLCLOCK - t ;
    printf ("nthreads %d, n: %ld, time: %g\n", nthreads, n, t) ;

    GB_mx_put_global (true) ;   
}

