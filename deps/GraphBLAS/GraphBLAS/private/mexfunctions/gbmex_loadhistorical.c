//------------------------------------------------------------------------------
// gbmex_loadhistorical: create a shallow GrB or GhB matrix for loadobj
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = gbmex_loadhistorical (S) creates a new GrB or GhB matrix for GraphBLAS
// 10.4.0 or later, from a struct created when a non-handle GrB matrix was
// saved to a *.mat file by GraphBLAS v10.3.1 or earlier.

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_WORK
#define FREE_WORK                   \
    GxB_Container_free (&Container) ;

#undef  FREE_ALL
#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&C) ;          \
    GrB_Matrix_free (&Y) ;

#define USAGE "usage: C = gbmex_loadhistorical (ghb, S)"

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

    GxB_Container Container = NULL ;
    GrB_Matrix *C_opaque = NULL, C = NULL, Y = NULL ;

    GBMX_USAGE (nargin == 2 && nargout == 1, USAGE) ;
    bool ghb = (bool) mxGetScalar (pargin [0]) ;
    int arena = ghb ? GrB_DEFAULT : MXARENA ;

    CHECK_ERROR (!mxIsStruct (pargin [1]), USAGE " where S is a struct") ;

    if (ghb) pargout [0] = gbmx_export_ghb_mxstruct (&C_opaque) ;

    struct gb_matrix_struct Matrix [1] ;
    gb_matrix matrix = &(Matrix [0]) ;

    //--------------------------------------------------------------------------
    // get the content of the GrB matrix from the struct
    //--------------------------------------------------------------------------

    gbmx_get_grb_matrix (matrix, pargin [1]) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // import the contents of the S struct into a new read-only GrB_Matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matlab_or_grb_matrix (&C, matrix, arena, err)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;

    OK (gb_export (C_opaque, &C, KIND_GRB, ghb, err)) ;
    ////////////////////////////////////////////////////////////////////////////
    if (!ghb)
    { 
        pargout [0] = gbmx_export_grb_mxstruct (&C) ;
    }

    gb_wrapup ( ) ;
}

