//------------------------------------------------------------------------------
// gbbuiltin: convert to a sparse or full built-in matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input may be either a GraphBLAS matrix struct or a standard built-in
// sparse or full matrix.  The output is a standard built-in sparse or full
// matrix: full if all entries are present, and sparse otherwise.

// Usage:

// A = gbbuiltin (X, type)

#include "gb_interface.h"

#define USAGE "usage: A = gbbuiltin (X, type)"

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
    // get the input matrix
    //--------------------------------------------------------------------------

    GrB_Matrix X = gb_get_shallow (pargin [0]) ;
    GrB_Type xtype ;
    OK (GxB_Matrix_type (&xtype, X)) ;

    //--------------------------------------------------------------------------
    // get the desired type, and typecast if needed
    //--------------------------------------------------------------------------

    GrB_Type type = gb_mxstring_to_type (pargin [1]) ;
    GrB_Matrix T = NULL ;
    if (type != xtype)
    { 
        T = gb_typecast (X, type, GxB_BY_COL, GxB_SPARSE + GxB_FULL) ;
        OK (GrB_Matrix_free (&X)) ;
        X = T ;
    }

    //--------------------------------------------------------------------------
    // export the input matrix to a built-in sparse or full matrix
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&X, KIND_BUILTIN) ;
    GB_WRAPUP ;
}

