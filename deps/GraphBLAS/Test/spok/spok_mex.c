#include "spok.h"
#include "mex.h"

/* -------------------------------------------------------------------------- */
/* SPOK mexFunction */
/* -------------------------------------------------------------------------- */

/* Copyright 2008-2011, Timothy A. Davis, http://suitesparse.com */
/* SPDX-License-Identifier: Apache-2.0 */

/* Checks the validity of a built-in sparse matrix.  Returns 1 if OK (or if the
   matrix is not sparse), 0 if the row indices are jumbled (C=A' might lead to
   a valid C matrix) or if the matrix includes explicit zero entries (which can
   be fixed with C=A*1).  Raises an error if the matrix is corrupted beyond
   repair.  Reports a warning if the matrix is not sparse (it is not checked),
   or if the matrix has jumbled row indices or explicit zeros.
 */

#define ERROR(s)   mexErrMsgIdAndTxt  ("SPOK:InvalidMatrix", s)
#define USAGE(s)   mexErrMsgIdAndTxt  ("SPOK:InvalidUsage", s)
#define WARNING(s) mexWarnMsgIdAndTxt ("SPOK:QuestionableMatrix", s)
#define LEN 200

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    mwIndex *Ap, *Ai ;
    double *Ax, *Az ;
    char *As ;
    SPOK_INT i, j, p, njumbled, nzeros, m, n, nzmax ;
    char msg [LEN+1] ;

    /* ---------------------------------------------------------------------- */
    /* check the usage */
    /* ---------------------------------------------------------------------- */

    if (nargout > 1)
    {
        USAGE ("too many output arguments") ;
    }
    if (nargin != 1)
    {
        USAGE ("usage: spok(A) where A is sparse") ;
    }

    if (!mxIsSparse (pargin [0]))
    {
        mexWarnMsgIdAndTxt ("SPOK:NotSparse", "non-sparse matrix not checked") ;
        pargout [0] = mxCreateDoubleScalar (1) ;
        return ;
    }

    Ap = (mwIndex *) mxGetJc (pargin [0]) ;
    Ai = (mwIndex *) mxGetIr (pargin [0]) ;
    m = mxGetM (pargin [0]) ;
    n = mxGetN (pargin [0]) ;
    nzmax = mxGetNzmax (pargin [0]) ;

    if (mxIsComplex (pargin [0]))
    {
        Az = mxGetPi (pargin [0]) ;
        if (Az == NULL)
        {
            ERROR ("complex, but with no imaginary part") ;
        }
    }
    else
    {
        Az = NULL ;
    }

    if (mxIsLogical (pargin [0]))
    {
        Ax = NULL ;
        As = (char *) mxGetData (pargin [0]) ;
        if (As == NULL)
        {
            ERROR ("logical, but with no values present") ;
        }
    }
    else
    {
        Ax = (double *) mxGetData (pargin [0]) ;
        As = NULL ;
        if (Ax == NULL)
        {
            ERROR ("double, but with no values present") ;
        }
    }

    switch (spok (m, n, nzmax, (int64_t *) Ap, (int64_t *) Ai, Ax, Az, As,
        &njumbled, &nzeros))
    {

        case SPOK_FATAL_M:
            ERROR ("negative number of rows") ;

        case SPOK_FATAL_N:
            ERROR ("negative number of columns") ;

        case SPOK_FATAL_NZMAX:
            ERROR ("nzmax(A) invalid") ;

        case SPOK_FATAL_P:
            ERROR ("column pointers invalid") ;

        case SPOK_FATAL_I:
            ERROR ("row indices out of range") ;

        case SPOK_WARNING:
            sprintf (msg, "%g jumbled row indices, %g explicit zeros",
                (double) njumbled, (double) nzeros) ;
            WARNING (msg) ;
            pargout [0] = mxCreateDoubleScalar (0) ;
            return ;

        case SPOK_OK:
            pargout [0] = mxCreateDoubleScalar (1) ;
            return ;

    }
}
