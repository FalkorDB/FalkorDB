//------------------------------------------------------------------------------
// gbdegree: number of entries in each vector of a GraphBLAS matrix struct
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input may be either a GraphBLAS matrix struct or a standard built-in
// sparse matrix.

//  gbdegree (A, 'row')     row degree
//  gbdegree (A, 'col')     column degree

#include "gb_interface.h"

#define USAGE "usage: degree = gbdegree (A, dim)"

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

    GrB_Matrix d = NULL, x = NULL ;
    GrB_Matrix A = gb_get_shallow (pargin [0]) ;
    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;

    #define LEN 256
    char dim_string [LEN+2] ;
    gb_mxstring_to_string (dim_string, LEN, pargin [1], "dim") ;

    //--------------------------------------------------------------------------
    // compute the row/column degree
    //--------------------------------------------------------------------------

    if (MATCH (dim_string, "row"))
    {

        //----------------------------------------------------------------------
        // row degree
        //----------------------------------------------------------------------

        // x = ones (ncols,1) ;
        OK (GrB_Matrix_new (&x, GrB_INT64, ncols, 1)) ;
        OK (GrB_Matrix_assign_INT64 (x, NULL, NULL, 1, GrB_ALL, ncols,
            GrB_ALL, 1, NULL)) ;
        // d = A*x using the PLUS_PAIR semiring
        OK (GrB_Matrix_new (&d, GrB_INT64, nrows, 1)) ;
        OK (GrB_mxm (d, NULL, NULL, GxB_PLUS_PAIR_INT64, A, x, NULL)) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // column degree
        //----------------------------------------------------------------------

        // x = ones (nrows,1) ;
        OK (GrB_Matrix_new (&x, GrB_INT64, nrows, 1)) ;
        OK (GrB_Matrix_assign_INT64 (x, NULL, NULL, 1, GrB_ALL, nrows,
            GrB_ALL, 1, NULL)) ;
        // d = A'*x using the PLUS_PAIR semiring
        OK (GrB_Matrix_new (&d, GrB_INT64, ncols, 1)) ;
        OK (GrB_mxm (d, NULL, NULL, GxB_PLUS_PAIR_INT64, A, x, GrB_DESC_T0)) ;
    }

    OK (GrB_Matrix_free (&x)) ;
    OK (GrB_Matrix_free (&A)) ;
    pargout [0] = gb_export ((GrB_Matrix *) &d, KIND_GRB) ;
    GB_WRAPUP ;
}

