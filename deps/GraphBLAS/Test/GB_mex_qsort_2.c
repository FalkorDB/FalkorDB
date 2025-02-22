//------------------------------------------------------------------------------
// GB_mex_qsort_2: sort using GB_qsort_2
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

    bool J_is_32 ;
    if (mxIsClass (pargin [1], "uint32"))
    { 
        J_is_32 = true ;
    }
    else if (mxIsClass (pargin [1], "uint64"))
    {
        J_is_32 = false ;
    }
    else
    {
        mexErrMsgTxt ("J must be a uint32 or uint64 array") ;
    }

    void *I = mxGetData (pargin [0]) ;
    int64_t n = (uint64_t) mxGetNumberOfElements (pargin [0]) ;

    void *J = mxGetData (pargin [1]) ;
    if (n != (uint64_t) mxGetNumberOfElements (pargin [1])) 
    {
        mexErrMsgTxt ("I and J must be the same length") ;
    }

    pargout [0] = GB_mx_create_full (n, 1, I_is_32 ? GrB_UINT32 : GrB_UINT64) ;
    void *Iout = mxGetData (pargout [0]) ;
    memcpy (Iout, I, n * (I_is_32 ? sizeof (uint32_t) : sizeof (uint64_t))) ;

    pargout [1] = GB_mx_create_full (n, 1, J_is_32 ? GrB_UINT32 : GrB_UINT64) ;
    void *Jout = mxGetData (pargout [1]) ;
    memcpy (Jout, J, n * (J_is_32 ? sizeof (uint32_t) : sizeof (uint64_t))) ;

//  double t = GB_omp_get_wtime ( ) ;
    GB_qsort_2 (Iout, I_is_32, Jout, J_is_32, n) ;
//  printf ("qsort_2 time %g\n", GB_omp_get_wtime ( ) - t) ;
}

