//------------------------------------------------------------------------------
// gbmex_unopinfo : print a GraphBLAS unary op (for illustration only)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbmex_unopinfo (unop)
// gbmex_unopinfo (unop, type)
// ok = gbmex_unopinfo (unop)

#include "gb_interface.h"
#include "gb_string_to_unop.c"
#include "gb_string_and_type_to_unop.c"

#include "gbmx_interface.h"

#define USAGE "usage: GrB.unopinfo (unop) or GrB.unopinfo (unop,type)"

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

    GrB_Type type = NULL ;  // use default type if NULL
    GrB_UnaryOp op = NULL ;

    GBMX_USAGE (nargin >= 1 && nargin <= 2 && nargout <= 1, USAGE) ;

    if (nargout == 1)
    { 
        pargout [0] = mxCreateLogicalScalar (true) ;
    }

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    char op_string [LEN+2] ;
    char type_string [LEN+2] ;
    gbmx_mxstring_to_string (op_string, LEN, pargin [0], "unary operator") ;
    if (nargin > 1)
    { 
        gbmx_mxstring_to_string (type_string, LEN, pargin [1], "type") ;
    }

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // construct the GraphBLAS unary operator and print it
    //--------------------------------------------------------------------------

    if (nargin > 1)
    { 
        type = gb_string_to_type (type_string) ;
    }

    OK (gb_string_to_unop (&op, op_string, type, err)) ;
    CHECK_ERROR (op == NULL, "unknown operator") ;

    int pr = (nargout < 1) ? GxB_COMPLETE : GxB_SILENT ;
    OK (GxB_UnaryOp_fprint (op, op_string, pr, NULL)) ;
    gb_wrapup ( ) ;
}

