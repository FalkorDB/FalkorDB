//------------------------------------------------------------------------------
// gbmex_disp: display a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbmex_disp (C, level)

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_ALL
#define FREE_ALL GrB_Matrix_free (&C_to_free) ;

#define USAGE "usage: gbmex_disp (C, level)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs (no outputs to construct)
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, C_to_free = NULL ;

    GBMX_USAGE (nargin == 2 && nargout == 0, USAGE) ;
    int arena = GrB_DEFAULT ;   // use default arena for temporary workspace

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;
    int level = (int) mxGetScalar (pargin [1]) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the input matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&C, &C_to_free, &(Matrix [0]), arena, err)) ;

    //--------------------------------------------------------------------------
    // print the matrix
    //--------------------------------------------------------------------------

    // print 1-based indices
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, true, GxB_PRINT_1BASED)) ;

    // print sizes of shallow components
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, true,
        GxB_INCLUDE_READONLY_STATISTICS)) ;

    char *name ;

    if (Matrix [0].kind == KIND_GHB)
    { 
        // C is a GhB handle matrix object
        name = "GhB matrix" ;
    }
    else if (Matrix [0].kind == KIND_GRB)
    { 
        // C is a GrB value matrix object
        name = "GrB matrix" ;
    }
    else
    { 
        // C is a shallow GrB_Matrix that holds a builtin MATLAB/Octave matrix
        #ifdef OCTAVE
        name = "Octave matrix" ;
        #else
        name = "MATLAB matrix" ;
        #endif
    }

    OK (GxB_Matrix_fprint (C, name, level, NULL)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_ALL ;
    gb_wrapup ( ) ;
}

