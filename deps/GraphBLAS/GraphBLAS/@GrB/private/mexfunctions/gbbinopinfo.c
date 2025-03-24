//------------------------------------------------------------------------------
// gbbinopinfo : print a GraphBLAS binary op (for illustration only)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbbinopinfo (binop)
// gbbinopinfo (binop, type)
// ok = gbbinopinfo (binop)

#include "gb_interface.h"

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
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin >= 1 && nargin <= 2 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // construct the GraphBLAS binary operator and print it
    //--------------------------------------------------------------------------

    #define LEN 256
    char opstring [LEN+2] ;
    gb_mxstring_to_string (opstring, LEN, pargin [0], "binary operator") ;

    GrB_Type type = NULL ;
    if (nargin > 1)
    { 
        type = gb_mxstring_to_type (pargin [1]) ;
        CHECK_ERROR (type == NULL, "unknown type") ;
    }

    GrB_BinaryOp op2 = NULL ;
    GrB_IndexUnaryOp idxunop = NULL ;
    int64_t ithunk = 0 ;

    gb_mxstring_to_binop_or_idxunop (pargin [0], type, type,
        &op2, &idxunop, &ithunk) ;

    int pr = (nargout < 1) ? GxB_COMPLETE : GxB_SILENT ;
    if (idxunop != NULL)
    {
        OK (GxB_IndexUnaryOp_fprint (idxunop, opstring, pr, NULL)) ;
    }
    else
    {
        OK (GxB_BinaryOp_fprint (op2, opstring, pr, NULL)) ;
    }
    if (nargout == 1)
    {
        pargout [0] = mxCreateLogicalScalar (true) ;
    }
    gb_wrapup ( ) ;
}

