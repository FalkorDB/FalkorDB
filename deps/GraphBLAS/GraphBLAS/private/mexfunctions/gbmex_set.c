//------------------------------------------------------------------------------
// gbmex_set: set a property of a GrB_Matrix (GhB interface only)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The matrix can only be a GraphBLAS GhB matrix.

#include "gb_interface.h"
#include "gbmx_interface.h"

#define USAGE "usage: gbmex_set (C, state, value)"

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

    GrB_Matrix C = NULL ;
    int arena = GrB_DEFAULT ;   // use default arena for all GhB matrices

    GBMX_USAGE (nargin == 3 && nargout == 0, USAGE) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;

    char state [LEN+2], char_value [LEN+2] ;
    gbmx_mxstring_to_string (state, LEN, pargin [1], "state") ;

    char_value [0] = '\0' ;
    int32_t int_value = 0 ;
    bool value_ok = mxIsLogical (pargin [2]) || mxIsNumeric (pargin [2]) ;
    if (mxIsChar (pargin [2]))
    { 
        gbmx_mxstring_to_string (char_value, LEN, pargin [2], "value") ;
    }
    else if (value_ok)
    { 
        int_value = (int32_t) mxGetScalar (pargin [2]) ;
    }
    bool is_32_or_64 = value_ok && (int_value == 32 || int_value == 64) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the input matrix
    //--------------------------------------------------------------------------

    C = (Matrix [0]).G ;
    CHECK_ERROR (C == NULL, "input matrix must be a GhB matrix") ; 

    //--------------------------------------------------------------------------
    // set the state
    //--------------------------------------------------------------------------

    if (MATCH (state, "format"))
    { 
        // set the format and/or sparsity
        bool fmt_present = false, sparsity_present = false ;
        int sparsity = 0, fmt = 0 ;
        bool ok = gb_string_to_format (char_value, &fmt, &fmt_present,
            &sparsity, &sparsity_present) ;
        CHECK_ERROR (!ok, "invalid format") ;
        if (fmt_present)
        { 
            // set the format: by row or by col
            OK (GrB_Matrix_set_INT32 (C, fmt, GxB_FORMAT)) ;
        }
        if (sparsity_present)
        { 
            // set the sparsity: sparse/hypersparse/bitmap/full
            OK (GrB_Matrix_set_INT32 (C, sparsity, GxB_SPARSITY_CONTROL)) ;
        }
    }
    else if (MATCH (state, "iso"))
    { 
        // set the iso state
        CHECK_ERROR (!value_ok, "invalid iso value") ;
        OK (GrB_Matrix_set_INT32 (C, int_value, GxB_ISO)) ;
    }
    else 
    { 
        // set an integer state: offset, row, or column
        CHECK_ERROR (!is_32_or_64, "invalid value (must be 32 or 64)") ;
        if (MATCH (state, "offset"))
        { 
            // set the integer sizes for offsets
            OK (GrB_Matrix_set_INT32 (C, int_value, GxB_OFFSET_INTEGER_HINT)) ;
        }
        else if (MATCH (state, "column") || MATCH (state, "col"))
        { 
            // set the integer sizes for column indices
            OK (GrB_Matrix_set_INT32 (C, int_value, GxB_COLINDEX_INTEGER_HINT)) ;
        }
        else if (MATCH (state, "row"))
        { 
            // set the integer sizes for row indices
            OK (GrB_Matrix_set_INT32 (C, int_value, GxB_ROWINDEX_INTEGER_HINT)) ;
        }
        else
        { 
            ERROR ("invalid state (must be 'format', 'iso', 'offset', "
                "'row', or 'col')", GrB_INVALID_VALUE) ;
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    FREE_ALL ;
    gb_wrapup ( ) ;
}

