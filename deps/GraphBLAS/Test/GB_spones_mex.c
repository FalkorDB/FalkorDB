//------------------------------------------------------------------------------
// GB_spones_mex: like spones(A) in MATLAB but do not drop zeros on input
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// The MATLAB built-in function spones(A) has changed, as of MATLAB R2019b.
// It now drops zeros on input.  Prior versions converted them to 1 on output.
// The tests here use the old behavior, so this function replaces spones(A)
// with GB_spones_mex (A), which has the old behavior of spones.

#include "mex.h"
#include <string.h>

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

    if (nargin != 1 || nargout > 1 || !mxIsSparse (pargin [0]))
    {
        mexErrMsgTxt ("usage: C = GB_spones_mex (A), A must be sparse") ;
    }

    //--------------------------------------------------------------------------
    // get the input matrix
    //--------------------------------------------------------------------------

    mwSize m = mxGetM (pargin [0]) ;
    mwSize n = mxGetN (pargin [0]) ;
    mwIndex *Ap = mxGetJc (pargin [0]) ;
    mwIndex *Ai = mxGetIr (pargin [0]) ;
    mwSize nz = Ap [n] ;

    //--------------------------------------------------------------------------
    // create the output matrix
    //--------------------------------------------------------------------------

    pargout [0] = mxCreateSparse (m, n, nz+1, mxREAL) ;
    mwIndex *Cp = mxGetJc (pargout [0]) ;
    mwIndex *Ci = mxGetIr (pargout [0]) ;
    double *Cx = mxGetDoubles (pargout [0]) ;

    memcpy (Cp, Ap, (n+1) * sizeof (mwIndex)) ;
    memcpy (Ci, Ai, nz    * sizeof (mwIndex)) ;
    for (mwSize p = 0 ; p < nz ; p++)
    {
        Cx [p] = 1 ;
    }
}

