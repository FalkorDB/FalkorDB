//------------------------------------------------------------------------------
// gbmex_degree: number of entries in each vector of a GraphBLAS matrix struct
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input may be either a GraphBLAS matrix struct or a standard built-in
// sparse matrix.

//  gbmex_degree (ghb, A, 'row')     row degree
//  gbmex_degree (ghb, A, 'col')     column degree

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_WORK
#define FREE_WORK                   \
    GrB_Matrix_free (&x) ;          \
    GrB_Matrix_free (&A_to_free) ;

#undef  FREE_ALL
#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&d) ;

#define USAGE "usage: degree = gbmex_degree (ghb, A, dim)"

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

    GrB_Matrix *d_opaque = NULL, d = NULL, x = NULL, A = NULL,
        A_to_free = NULL ;

    GBMX_USAGE (nargin == 3 && nargout <= 1, USAGE) ;
    bool ghb = (bool) mxGetScalar (pargin [0]) ;
    int arena = ghb ? GrB_DEFAULT : MXARENA ;

    if (ghb) pargout [0] = gbmx_export_ghb_mxstruct (&d_opaque) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [1]) ;

    char dim_string [LEN+2] ;
    gbmx_mxstring_to_string (dim_string, LEN, pargin [2], "dim") ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get input matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;

    uint64_t nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;

    //--------------------------------------------------------------------------
    // compute the row/column degree
    //--------------------------------------------------------------------------

    if (MATCH (dim_string, "row"))
    { 

        //----------------------------------------------------------------------
        // row degree
        //----------------------------------------------------------------------

        // x = ones (ncols,1) ;
        OK (GxB_Matrix_new_arena (&x, GrB_INT64, ncols, 1, arena, arena)) ;
        OK (GrB_Matrix_assign_INT64 (x, NULL, NULL, 1, GrB_ALL, ncols,
            GrB_ALL, 1, NULL)) ;
        // d = A*x using the PLUS_PAIR semiring
        OK (GxB_Matrix_new_arena (&d, GrB_INT64, nrows, 1, arena, arena)) ;
        OK (GrB_mxm (d, NULL, NULL, GxB_PLUS_PAIR_INT64, A, x, NULL)) ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // column degree
        //----------------------------------------------------------------------

        // x = ones (nrows,1) ;
        OK (GxB_Matrix_new_arena (&x, GrB_INT64, nrows, 1, arena, arena)) ;
        OK (GrB_Matrix_assign_INT64 (x, NULL, NULL, 1, GrB_ALL, nrows,
            GrB_ALL, 1, NULL)) ;
        // d = A'*x using the PLUS_PAIR semiring
        OK (GxB_Matrix_new_arena (&d, GrB_INT64, ncols, 1, arena, arena)) ;
        OK (GrB_mxm (d, NULL, NULL, GxB_PLUS_PAIR_INT64, A, x, GrB_DESC_T0)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;

    OK (gb_export (d_opaque, (GrB_Matrix *) &d, KIND_GRB, ghb, err)) ;
    ////////////////////////////////////////////////////////////////////////////
    if (!ghb)
    { 
        pargout [0] = gbmx_export_grb_mxstruct ((GrB_Matrix *) &d) ;
    }

    gb_wrapup ( ) ;
}

