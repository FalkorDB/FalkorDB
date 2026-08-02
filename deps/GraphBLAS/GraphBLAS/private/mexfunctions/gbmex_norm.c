//------------------------------------------------------------------------------
// gbmex_norm: norm (A,kind)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"
#include "gb_norm.c"

#include "gbmx_interface.h"

#undef  FREE_ALL
#define FREE_ALL GrB_Matrix_free (&A_to_free) ;

#define USAGE "usage: s = gbmex_norm (A, kind)"

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

    GrB_Matrix A = NULL, A_to_free = NULL ;
    int arena = GrB_DEFAULT ;   // use default arena for temporary workspace

    GBMX_USAGE (nargin == 2 && nargout <= 1, USAGE) ;

    pargout [0] = mxCreateDoubleScalar (0) ;
    double *s_output = (double *) mxGetData (pargout [0]) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;

    int64_t norm_kind = gbmx_norm_kind (pargin [1]) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the input matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;

    GrB_Type atype ;
    OK (GxB_Matrix_type (&atype, A)) ;

    uint64_t anrows, ancols ;
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;

    int sparsity ;
    OK (GrB_Matrix_get_INT32 (A, &sparsity, GxB_SPARSITY_STATUS)) ;

    //--------------------------------------------------------------------------
    // s = norm (A,kind)
    //--------------------------------------------------------------------------

    double s ;

    bool is_dense ;
    OK (gb_is_dense (&is_dense, A, err)) ;

    if (norm_kind == INT64_MIN && !is_dense)
    { 
        // norm (A,-inf) is zero if A is not full
        s = 0 ;
    }
    else if ((atype == GrB_FP32 || atype == GrB_FP64)
        && (sparsity != GxB_BITMAP)
        && (anrows == 1 || ancols == 1 || norm_kind == 0))
    { 
        // s = norm (A,p) where A is an FP32 or FP64 vector,
        // or when p = 0 (for Frobenius norm).  A cannot be bitmap.
        uint64_t anz ;
        OK (GrB_Matrix_nvals (&anz, A)) ;
        OK (GB_helper10 (&s, A->x, A->iso, NULL, false, atype,
            norm_kind, anz)) ;
        CHECK_ERROR (s < 0, "unknown norm") ;
    }
    else
    { 
        // s = norm (A, norm_kind)
        OK (gb_norm (&s, A, norm_kind, arena, err)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_ALL ;
    (*s_output) = s ;
    gb_wrapup ( ) ;
}

