//------------------------------------------------------------------------------
// gbmex_normdiff: norm (A-B,kind)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This function accesses opaque content and GB_methods inside GraphBLAS.

#include "gb_interface.h"
#include "gb_norm.c"

#include "gbmx_interface.h"

#undef  FREE_ALL
#define FREE_ALL                    \
    GrB_Matrix_free (&A_to_free) ;  \
    GrB_Matrix_free (&B_to_free) ;  \
    GrB_Matrix_free (&X) ;

#define USAGE "usage: s = gbmex_normdiff (A, B, kind)"

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

    GrB_Matrix A = NULL, B = NULL, X = NULL, A_to_free = NULL,
        B_to_free = NULL ;
    int arena = GrB_DEFAULT ;   // use default arena for temporary workspace

    GBMX_USAGE (nargin == 3 && nargout <= 1, USAGE) ;

    pargout [0] = mxCreateDoubleScalar (0) ;
    double *s_output = (double *) mxGetData (pargout [0]) ;

    //--------------------------------------------------------------------------
    // get the inputs 
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [2] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;
    gbmx_get_matrix (&(Matrix [1]), pargin [1]) ;

    int64_t norm_kind = gbmx_norm_kind (pargin [2]) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the inputs 
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;
    OK (gb_get_matrix (&B, &B_to_free, &(Matrix [1]), arena, err)) ;

    GrB_Type atype, btype ;
    OK (GxB_Matrix_type (&atype, A)) ;
    OK (GxB_Matrix_type (&btype, B)) ;

    uint64_t anrows, ancols, bnrows, bncols ;
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;
    OK (GrB_Matrix_nrows (&bnrows, B)) ;
    OK (GrB_Matrix_ncols (&bncols, B)) ;
    if (anrows != bnrows || ancols != bncols)
    { 
        ERROR ("A and B must have the same size", GrB_DIMENSION_MISMATCH) ;
    }

    //--------------------------------------------------------------------------
    // s = norm (A-B,kind)
    //--------------------------------------------------------------------------

    double s ;

    bool A_is_dense, B_is_dense ;
    OK (gb_is_dense (&A_is_dense, A, err)) ;
    OK (gb_is_dense (&B_is_dense, A, err)) ;

    if (A_is_dense && B_is_dense &&
        (atype == GrB_FP32 || atype == GrB_FP64) && (atype == btype)
        && (anrows == 1 || ancols == 1 || norm_kind == 0))
    { 
        // s = norm (A-B,p) where A and B are full FP32 or FP64 vectors,
        // or when p = 0 (for Frobenius norm)
        uint64_t anz ;
        OK (GrB_Matrix_nvals (&anz, A)) ;
        OK (GB_helper10 (&s, A->x, A->iso, B->x, B->iso, atype,
            norm_kind, anz)) ;
        CHECK_ERROR (s < 0, "unknown norm") ;
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
        OK (GxB_Matrix_new_arena (&X, xtype, anrows, ancols, arena, arena)) ;
        OK1 (X, GrB_Matrix_eWiseAdd_BinaryOp (X, NULL, NULL, op, A, B, NULL)) ;

        // s = norm (X, norm_kind)
        OK (gb_norm (&s, X, norm_kind, arena, err)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_ALL ;
    (*s_output) = s ;
    gb_wrapup ( ) ;
}

