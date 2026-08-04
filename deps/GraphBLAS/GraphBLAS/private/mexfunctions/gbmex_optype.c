//------------------------------------------------------------------------------
// gbmex_optype : determine the type of a binary operator from the input types
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// optype = gbmex_optype (atype, btype)

#include "gb_interface.h"
#include "gbmx_interface.h"

#define USAGE "usage: c = GrB.optype (atype, btype)"

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

    GBMX_USAGE (nargin == 2 && nargout <= 1, USAGE) ;

    char atype_string [LEN+2] ;
    char btype_string [LEN+2] ;
    gbmx_mxstring_to_string (atype_string, LEN, pargin [0], "atype") ;
    gbmx_mxstring_to_string (btype_string, LEN, pargin [1], "btype") ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get atype and btype
    //--------------------------------------------------------------------------

    GrB_Type atype = gb_string_to_type (atype_string) ;
    GrB_Type btype = gb_string_to_type (btype_string) ;

    //--------------------------------------------------------------------------
    // determine the optype
    //--------------------------------------------------------------------------

    GrB_Type optype = gb_default_type (atype, btype) ;
    CHECK_ERROR (optype == NULL, "unknown type") ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // return result as a MATLAB string
    //--------------------------------------------------------------------------

    pargout [0] = gbmx_type_to_mxstring (optype) ;
    gb_wrapup ( ) ;
}

