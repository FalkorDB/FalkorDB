//------------------------------------------------------------------------------
// gbbandwidth: compute the lower and/or upper bandwidth of a GrB matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// usage:

// [lo,hi] = gbbandwidth (A, compute_lo, compute_hi)

#include "gb_interface.h"

#define USAGE "usage: [lo,hi] = gbbandwidth (A, compute_lo, compute_hi)"

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

    gb_usage (nargin == 3 && nargout == 2, USAGE) ;
    GrB_Matrix A = gb_get_shallow (pargin [0]) ;
    bool compute_lo = (bool) mxGetScalar (pargin [1]) ;
    bool compute_hi = (bool) mxGetScalar (pargin [2]) ;
    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;

    //--------------------------------------------------------------------------
    // compute lo and hi
    //--------------------------------------------------------------------------

    int64_t hi = 0, lo = 0 ;
    GrB_Matrix x = NULL, imin = NULL, imax = NULL, idiag = NULL ;

    GxB_Format_Value fmt ;
    OK (GxB_Matrix_Option_get (A, GxB_FORMAT, &fmt)) ;
    bool by_col = (fmt == GxB_BY_COL) ;

    if (by_col)
    { 

        //----------------------------------------------------------------------
        // A is held by column
        //----------------------------------------------------------------------

        OK (GrB_Matrix_new (&x, GrB_BOOL, 1, nrows)) ;
        OK (GrB_Matrix_new (&imin, GrB_INT64, 1, ncols)) ;
        OK (GrB_Matrix_new (&imax, GrB_INT64, 1, ncols)) ;
        OK (GrB_Matrix_new (&idiag, GrB_INT64, 1, ncols)) ;

        // x = true (1, nrows)
        OK (GrB_Matrix_assign_BOOL (x, NULL, NULL, true, GrB_ALL, 1, GrB_ALL,
            nrows, NULL)) ;

        if (compute_hi)
        { 
            // imin = x*A, where imin(j) = min row index in column j
            OK (GrB_mxm (imin, NULL, NULL, GxB_MIN_FIRSTJ_INT64, x, A, NULL)) ;
        }

        if (compute_lo)
        { 
            // imax = x*A, where imax(j) = max row index in column j
            OK (GrB_mxm (imax, NULL, NULL, GxB_MAX_FIRSTJ_INT64, x, A, NULL)) ;
        }

        // construct idiag: idiag(j) = j with same sparsity pattern as imin/imax
        OK (GrB_Matrix_apply_IndexOp_INT64 (idiag, NULL, NULL,
            GrB_COLINDEX_INT64, compute_hi ? imin : imax, 0, NULL)) ;

        if (compute_hi)
        { 
            // imin = idiag - imin
            OK (GrB_Matrix_eWiseMult_BinaryOp (imin, NULL, NULL,
                GrB_MINUS_INT64, idiag, imin, NULL)) ;
            // hi = max (imin, 0) ;
            OK (GrB_Matrix_reduce_INT64 (&hi, GrB_MAX_INT64,
                GrB_MAX_MONOID_INT64, imin, NULL)) ;
        }

        if (compute_lo)
        { 
            // imax = imax - idiag
            OK (GrB_Matrix_eWiseMult_BinaryOp (imax, NULL, NULL,
                GrB_MINUS_INT64, imax, idiag, NULL)) ;
            // lo = max (imax, 0) ;
            OK (GrB_Matrix_reduce_INT64 (&lo, GrB_MAX_INT64,
                GrB_MAX_MONOID_INT64, imax, NULL)) ;
        }

    }
    else
    { 

        //----------------------------------------------------------------------
        // A is held by row
        //----------------------------------------------------------------------

        OK (GrB_Matrix_new (&x, GrB_BOOL, ncols, 1)) ;
        OK (GrB_Matrix_new (&imin, GrB_INT64, nrows, 1)) ;
        OK (GrB_Matrix_new (&imax, GrB_INT64, nrows, 1)) ;
        OK (GrB_Matrix_new (&idiag, GrB_INT64, nrows, 1)) ;

        // x = true (ncols, 1)
        OK (GrB_Matrix_assign_BOOL (x, NULL, NULL, true, GrB_ALL, ncols,
            GrB_ALL, 1, NULL)) ;

        if (compute_lo)
        { 
            // imin = A*x, where imin(i) = min column index in row i
            OK (GrB_mxm (imin, NULL, NULL, GxB_MIN_FIRSTJ_INT64, A, x, NULL)) ;
        }

        if (compute_hi)
        { 
            // imax = A*x, where imax(i) = max column index in row i
            OK (GrB_mxm (imax, NULL, NULL, GxB_MAX_FIRSTJ_INT64, A, x, NULL)) ;
        }

        // construct idiag: idiag(i) = i with same sparsity pattern as imin/imax
        OK (GrB_Matrix_apply_IndexOp_INT64 (idiag, NULL, NULL,
            GrB_ROWINDEX_INT64, compute_lo ? imin : imax, 0, NULL)) ;

        if (compute_lo)
        { 
            // imin = idiag - imin
            OK (GrB_Matrix_eWiseMult_BinaryOp (imin, NULL, NULL,
                GrB_MINUS_INT64, idiag, imin, NULL)) ;
            // lo = max (imin, 0) ;
            OK (GrB_Matrix_reduce_INT64 (&lo, GrB_MAX_INT64,
                GrB_MAX_MONOID_INT64, imin, NULL)) ;
        }

        if (compute_hi)
        { 
            // imax = imax - idiag
            OK (GrB_Matrix_eWiseMult_BinaryOp (imax, NULL, NULL,
                GrB_MINUS_INT64, imax, idiag, NULL)) ;
            // hi = max (imax, 0) ;
            OK (GrB_Matrix_reduce_INT64 (&hi, GrB_MAX_INT64,
                GrB_MAX_MONOID_INT64, imax, NULL)) ;
        }
    }

    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Matrix_free (&x)) ;
    OK (GrB_Matrix_free (&idiag)) ;
    OK (GrB_Matrix_free (&imin)) ;
    OK (GrB_Matrix_free (&imax)) ;

    //--------------------------------------------------------------------------
    // return result as int64 scalars
    //--------------------------------------------------------------------------

    if (lo > FLINTMAX || hi > FLINTMAX)
    { 
        // output is int64 to avoid flint overflow
        int64_t *p ;
        pargout [0] = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL) ;
        // use mxGetData (best for Octave, fine for MATLAB)
        p = (int64_t *) mxGetData (pargout [0]) ;
        p [0] = (int64_t) lo ;
        pargout [1] = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL) ;
        p = (int64_t *) mxGetData (pargout [1]) ;
        p [0] = (int64_t) hi ;
    }
    else
    { 
        // output is double
        pargout [0] = mxCreateDoubleScalar ((double) lo) ;
        pargout [1] = mxCreateDoubleScalar ((double) hi) ;
    }

    GB_WRAPUP ;
}

