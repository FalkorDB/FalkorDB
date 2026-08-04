//------------------------------------------------------------------------------
// gbmex_type: type of a GraphBLAS or built-in matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage

// type = gbmex_type (A)

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_ALL
#define FREE_ALL GrB_Matrix_free (&A_to_free) ;

#define USAGE "usage: type = gbmex_type (A)"

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

    GrB_Matrix A = NULL, A_to_free = NULL ;
    int arena = GrB_DEFAULT ;   // use default arena for temporary workspace

    GBMX_USAGE (nargin == 1 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get the type of the input
    //--------------------------------------------------------------------------

    mxClassID class = mxGetClassID (pargin [0]) ;
    if (class == mxCELL_CLASS)
    { 
        pargout [0] = mxCreateString ("cell") ;
    }
    else if (class == mxCHAR_CLASS)
    { 
        pargout [0] = mxCreateString ("char") ;
    }
    else
    { 

        //----------------------------------------------------------------------
        // get inputs
        //----------------------------------------------------------------------

        struct gb_matrix_struct Matrix [1] ;
        gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;

        ////////////////////////////////////////////////////////////////////////

        //----------------------------------------------------------------------
        // get the input matrix properties
        //----------------------------------------------------------------------

        OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;

        GrB_Type type ;
        OK (GxB_Matrix_type (&type, A)) ;

        // The input matrix is freed, so that mx* methods can allocate memory
        // below.  This eliminates any potential memory leaks if A is a handle
        // GrB matrix using malloc/free.

        FREE_ALL ;

        ////////////////////////////////////////////////////////////////////////

        //----------------------------------------------------------------------
        // return the type
        //----------------------------------------------------------------------

        pargout [0] = gbmx_type_to_mxstring (type) ;
    }

    gb_wrapup ( ) ;
}

