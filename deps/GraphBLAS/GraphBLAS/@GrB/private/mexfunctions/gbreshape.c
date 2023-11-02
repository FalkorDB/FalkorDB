//------------------------------------------------------------------------------
// gbreshape: reshape a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// usage:

// C = gbreshape (A, nrows_new, ncols_new, by_col)

#include "gb_interface.h"

#define USAGE "usage: C = gbreshape (A, nrows_new, ncols_new, by_col)"

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

    gb_usage ((nargin == 3 || nargin == 4) && nargout == 1, USAGE) ;
    GrB_Matrix A = gb_get_shallow (pargin [0]) ;
    GrB_Index nrows_new = gb_mxget_uint64_scalar (pargin [1], "nrows_new") ;
    GrB_Index ncols_new = gb_mxget_uint64_scalar (pargin [2], "ncols_new") ;
    bool by_col = (nargin == 3) ? true : ((bool) mxGetScalar (pargin [3])) ;

    //--------------------------------------------------------------------------
    // reshape the matrix
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL ;
    OK (GxB_Matrix_reshapeDup (&C, A, by_col, nrows_new, ncols_new, NULL)) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    GrB_Matrix_free (&A) ;
    pargout [0] = gb_export (&C, KIND_GRB) ;
    GB_WRAPUP ;
}

