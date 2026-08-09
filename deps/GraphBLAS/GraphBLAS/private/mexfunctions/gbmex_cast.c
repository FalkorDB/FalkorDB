//------------------------------------------------------------------------------
// gbmex_cast: convert to a sparse or full built-in MATLAB matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input may be either a GraphBLAS matrix struct or a standard built-in
// MATLAB sparse or full matrix.  The output is a GhB matrix but with a data
// structure that matches a standard built-in MATLAB/Octave sparse or full
// matrix: full if all entries are present, and sparse otherwise.  The
// matrix is then typically passed to the gbmex_builtin mexFunction to
// construct a built-in MATLAB/Octave matrix.

// Usage:

// C = gbmex_cast (X, type)

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_WORK
#define FREE_WORK                   \
    GrB_Matrix_free (&X_to_free) ;

#undef  FREE_ALL
#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&C) ;

#define USAGE "usage: C = gbmex_cast (X, type)"

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

    GrB_Matrix *C_opaque = NULL, X = NULL, X_to_free = NULL, C = NULL ;

    GBMX_USAGE (nargin == 2 && nargout <= 1, USAGE) ;
    int arena = GrB_DEFAULT ;   // output is always GhB

    pargout [0] = gbmx_export_ghb_mxstruct (&C_opaque) ;

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;

    char type_string [LEN+2] ;
    gbmx_mxstring_to_string (type_string, LEN, pargin [1], "type") ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the input matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&X, &X_to_free, &(Matrix [0]), arena, err)) ;

    //--------------------------------------------------------------------------
    // make a deep copy and typecast to the desired type
    //--------------------------------------------------------------------------

    GrB_Type type = gb_string_to_type (type_string) ;
    OK (gb_typecast (&C, X, type, GxB_BY_COL, GxB_SPARSE + GxB_FULL, arena,
        err)) ;

    // GrB_Matrix_wait is not yet called; this is done by gb_export below.

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    OK (gb_export (C_opaque, &C, KIND_BUILTIN, true, err)) ;
    gb_wrapup ( ) ;
}

