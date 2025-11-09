#include "mex.h"
#include "../spok.h"

/* spok_invalid: returns an invalid sparse matrix to test SPOK. */

/* Copyright 2008-2011, Timothy A. Davis, http://suitesparse.com */
/* SPDX-License-Identifier: Apache-2.0 */

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    mwIndex *Ap ;
    mwIndex *Ai ;
    double *Ax ;
    int kind ;

    if (nargin == 0)
    {
        kind = 0 ;
    }
    else
    {
        kind = (int) mxGetScalar (pargin [0]) ;
    }

    if (kind < 0)
    {

        /* a sparse 2-by-2 matrix with invalid column pointers*/
        pargout [0] = mxCreateSparse (2, 2, 4, mxREAL) ;
        Ap = (mwIndex *) mxGetJc (pargout [0]) ;
        Ai = (mwIndex *) mxGetIr (pargout [0]) ;
        Ax = mxGetPr (pargout [0]) ;
        Ap [0] = 2 ;
        Ap [1] = 2 ;
        Ap [2] = 2 ;

    }
    else if (kind == 0)
    {

        /* a sparse 1-by-1 matrix with one explicit zero */
        pargout [0] = mxCreateSparse (1, 1, 1, mxREAL) ;
        Ap = (mwIndex *) mxGetJc (pargout [0]) ;
        Ai = (mwIndex *) mxGetIr (pargout [0]) ;
        Ax = mxGetPr (pargout [0]) ;
        Ap [0] = 0 ;
        Ap [1] = 1 ;
        Ai [0] = 0 ;
        Ax [0] = 0 ;

    }
    else
    {

        /* a sparse 2-by-2 matrix with jumbled row indices */
        pargout [0] = mxCreateSparse (2, 2, 4, mxREAL) ;
        Ap = (mwIndex *) mxGetJc (pargout [0]) ;
        Ai = (mwIndex *) mxGetIr (pargout [0]) ;
        Ax = mxGetPr (pargout [0]) ;
        Ap [0] = 0 ;
        Ap [1] = 2 ;
        Ap [2] = 2 ;
        Ai [0] = 1 ;
        Ai [1] = 0 ;
        Ax [0] = 1 ;
        Ax [1] = 2 ;

    }
}
