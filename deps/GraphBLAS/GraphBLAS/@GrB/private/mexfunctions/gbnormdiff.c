//------------------------------------------------------------------------------
// gbnormdiff: norm (A-B,kind)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

#define USAGE "usage: s = gbnormdiff (A, B, kind)"

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

    gb_usage (nargin == 3 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get the inputs 
    //--------------------------------------------------------------------------

    GrB_Matrix A = gb_get_shallow (pargin [0]) ;
    GrB_Matrix B = gb_get_shallow (pargin [1]) ;
    int64_t norm_kind = gb_norm_kind (pargin [2]) ;

    GrB_Type atype, btype ;
    OK (GxB_Matrix_type (&atype, A)) ;
    OK (GxB_Matrix_type (&btype, B)) ;

    GrB_Index anrows, ancols, bnrows, bncols ;
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;
    OK (GrB_Matrix_nrows (&bnrows, B)) ;
    OK (GrB_Matrix_ncols (&bncols, B)) ;
    if (anrows != bnrows || ancols != bncols)
    {
        ERROR ("A and B must have the same size") ;
    }

    //--------------------------------------------------------------------------
    // s = norm (A-B,kind)
    //--------------------------------------------------------------------------

    double s ;

    if (GB_is_dense (A) && GB_is_dense (B) &&
        (atype == GrB_FP32 || atype == GrB_FP64) && (atype == btype)
        && (anrows == 1 || ancols == 1 || norm_kind == 0))
    {
        // s = norm (A-B,p) where A and B are full FP32 or FP64 vectors,
        // or when p = 0 (for Frobenius norm)
        GrB_Index anz ;
        OK (GrB_Matrix_nvals (&anz, A)) ;
        s = GB_helper10 (A->x, A->iso, B->x, B->iso,
            atype, norm_kind, anz) ;
        if (s < 0) ERROR ("unknown norm") ;
    }
    else
    {
        GrB_Type xtype ;
        GrB_BinaryOp op ;
        if (atype == GrB_FP32 && atype == btype)
        { 
            // both A and B are single: use FP32
            xtype = GrB_FP32 ;
            op = GrB_MINUS_FP32 ;
        }
        else if (atype == GxB_FC32 && btype == GxB_FC32)
        { 
            // both A and B are single complex: use FC32
            xtype = GxB_FC32 ;
            op = GxB_MINUS_FC32 ;
        }
        else if (atype == GxB_FC64 || btype == GxB_FC64 ||
                 atype == GxB_FC32 || btype == GxB_FC32)
        { 
            // either A or B are any kind of complex: use FC64
            xtype = GxB_FC64 ;
            op = GxB_MINUS_FC64 ;
        }
        else
        { 
            // both A and B are real (any kind): use FP64
            xtype = GrB_FP64 ;
            op = GrB_MINUS_FP64 ;
        }

        // X = A-B
        GrB_Matrix X ;
        OK (GrB_Matrix_new (&X, xtype, anrows, ancols)) ;
        OK1 (X, GrB_Matrix_eWiseAdd_BinaryOp (X, NULL, NULL, op, A, B, NULL)) ;

        // s = norm (X, norm_kind)
        s = gb_norm (X, norm_kind) ;
        OK (GrB_Matrix_free (&X)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Matrix_free (&B)) ;
    pargout [0] = mxCreateDoubleScalar (s) ;
    GB_WRAPUP ;
}

