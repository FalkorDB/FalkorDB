//------------------------------------------------------------------------------
// gbmex_bandwidth: compute the lower and/or upper bandwidth of a GrB matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// usage:

// [lo,hi] = gbmex_bandwidth (A, compute_lo, compute_hi)

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_ALL
#define FREE_ALL                    \
    GrB_Matrix_free (&A_to_free) ;  \
    GrB_Matrix_free (&x) ;          \
    GrB_Matrix_free (&idiag) ;      \
    GrB_Matrix_free (&imin) ;       \
    GrB_Matrix_free (&imax) ;

#define USAGE "usage: [lo,hi] = gbmex_bandwidth (A, compute_lo, compute_hi)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs and construct outputs
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL, A_to_free = NULL, x = NULL, imin = NULL, imax = NULL,
        idiag = NULL ;
    int arena = GrB_DEFAULT ;   // use default arena for temporary workspace

    GBMX_USAGE (nargin == 3 && nargout == 2, USAGE) ;

    pargout [0] = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL) ;
    pargout [1] = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL) ;
    int64_t *lo_output = (int64_t *) mxGetData (pargout [0]) ;
    int64_t *hi_output = (int64_t *) mxGetData (pargout [1]) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;

    bool compute_lo = (bool) mxGetScalar (pargin [1]) ;
    bool compute_hi = (bool) mxGetScalar (pargin [2]) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get input matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;

    uint64_t nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;

    //--------------------------------------------------------------------------
    // compute lo and hi
    //--------------------------------------------------------------------------

    int64_t hi = 0, lo = 0 ;

    int fmt ;
    OK (GrB_Matrix_get_INT32 (A, &fmt, GxB_FORMAT)) ;
    bool by_col = (fmt == GxB_BY_COL) ;

    if (by_col)
    { 

        //----------------------------------------------------------------------
        // A is held by column
        //----------------------------------------------------------------------

        OK (GxB_Matrix_new_arena (&x, GrB_BOOL, 1, nrows, arena, arena)) ;
        OK (GxB_Matrix_new_arena (&imin, GrB_INT64, 1, ncols, arena, arena)) ;
        OK (GxB_Matrix_new_arena (&imax, GrB_INT64, 1, ncols, arena, arena)) ;
        OK (GxB_Matrix_new_arena (&idiag, GrB_INT64, 1, ncols, arena, arena)) ;

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

        OK (GxB_Matrix_new_arena (&x, GrB_BOOL, ncols, 1, arena, arena)) ;
        OK (GxB_Matrix_new_arena (&imin, GrB_INT64, nrows, 1, arena, arena)) ;
        OK (GxB_Matrix_new_arena (&imax, GrB_INT64, nrows, 1, arena, arena)) ;
        OK (GxB_Matrix_new_arena (&idiag, GrB_INT64, nrows, 1, arena, arena)) ;

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

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    FREE_ALL ;
    (*lo_output) = (int64_t) lo ;
    (*hi_output) = (int64_t) hi ;
    gb_wrapup ( ) ;
}

