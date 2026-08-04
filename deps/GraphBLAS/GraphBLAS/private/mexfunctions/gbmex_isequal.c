//------------------------------------------------------------------------------
// gbmex_isequal: isequal (A,B)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmex_isequal returns isequal(A,B) for two matrices A and B.

// Usage:

//  result = gbmex_isequal (A,B)

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_ALL
#define FREE_ALL                    \
    GrB_Matrix_free (&A_to_free) ;  \
    GrB_Matrix_free (&B_to_free) ;

#define USAGE "usage: s = GrB.isequal (A, B)"

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

    GrB_Matrix A = NULL, B = NULL, A_to_free = NULL, B_to_free = NULL ;
    int arena = GrB_DEFAULT ;   // use default arena for temporary workspace

    GBMX_USAGE (nargin == 2 && nargout <= 1, USAGE) ;

    pargout [0] = mxCreateLogicalScalar (false) ;
    bool *s_output = (bool *) mxGetData (pargout [0]) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [2] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;
    gbmx_get_matrix (&(Matrix [1]), pargin [1]) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the arguments
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;
    OK (gb_get_matrix (&B, &B_to_free, &(Matrix [1]), arena, err)) ;

    //--------------------------------------------------------------------------
    // check if they are equal
    //--------------------------------------------------------------------------

    bool is_equal ;
    OK (gb_is_equal (&is_equal, A, B, arena, err)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_ALL ;
    (*s_output) = is_equal ;
    gb_wrapup ( ) ;
}

