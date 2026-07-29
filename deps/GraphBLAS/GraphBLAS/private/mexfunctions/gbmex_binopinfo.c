//------------------------------------------------------------------------------
// gbmex_binopinfo : print a GraphBLAS binary op (for illustration only)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbmex_binopinfo (binop)
// gbmex_binopinfo (binop, type)
// ok = gbmex_binopinfo (binop)

#include "gb_interface.h"
#include "gb_string_and_type_to_binop_or_idxunop.c"
#include "gb_string_to_binop.c"
#include "gb_string_to_binop_or_idxunop.c"

#include "gbmx_interface.h"

#define USAGE "usage: GrB.binopinfo (binop) or GrB.binopinfo (binop,type)"

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

    GrB_Type type = NULL ;
    GrB_BinaryOp binop = NULL ;
    GrB_IndexUnaryOp idxunop = NULL ;

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
    gbmx_mxstring_to_string (op_string, LEN, pargin [0], "binary operator") ;
    if (nargin > 1)
    { 
        gbmx_mxstring_to_string (type_string, LEN, pargin [1], "type") ;
    }

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // construct the GraphBLAS binary operator or index unary op and print it
    //--------------------------------------------------------------------------

    int64_t ithunk = 0 ;
    if (nargin > 1)
    { 
        type = gb_string_to_type (type_string) ;
    }

    OK (gb_string_to_binop_or_idxunop (&binop, &idxunop, &ithunk,
        op_string, type, type, err)) ;

    int pr = (nargout < 1) ? GxB_COMPLETE : GxB_SILENT ;
    if (idxunop != NULL)
    { 
        OK (GxB_IndexUnaryOp_fprint (idxunop, op_string, pr, NULL)) ;
    }
    else
    { 
        OK (GxB_BinaryOp_fprint (binop, op_string, pr, NULL)) ;
    }

    gb_wrapup ( ) ;
}

