//------------------------------------------------------------------------------
// gbnorm: norm (A,kind)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

#define USAGE "usage: s = gbnorm (A, kind)"

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

    gb_usage (nargin == 2 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get the inputs 
    //--------------------------------------------------------------------------

    GrB_Matrix A = gb_get_shallow (pargin [0]) ;
    int64_t norm_kind = gb_norm_kind (pargin [1]) ;

    GrB_Type atype ;
    OK (GxB_Matrix_type (&atype, A)) ;

    GrB_Index anrows, ancols ;
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;

    int sparsity ;
    OK (GxB_Matrix_Option_get (A, GxB_SPARSITY_STATUS, &sparsity)) ;

    //--------------------------------------------------------------------------
    // s = norm (A,kind)
    //--------------------------------------------------------------------------

    double s ;

    if (norm_kind == INT64_MIN && !GB_is_dense (A))
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
        GrB_Index anz ;
        OK (GrB_Matrix_nvals (&anz, A)) ;
        s = GB_helper10 (A->x, A->iso, NULL, false, atype, norm_kind, anz) ;
        if (s < 0) ERROR ("unknown norm") ;
    }
    else
    { 
        // s = norm (A, norm_kind)
        s = gb_norm (A, norm_kind) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&A)) ;
    pargout [0] = mxCreateDoubleScalar (s) ;
    GB_WRAPUP ;
}

