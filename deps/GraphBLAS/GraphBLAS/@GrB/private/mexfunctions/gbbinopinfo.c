//------------------------------------------------------------------------------
// gbbinopinfo : print a GraphBLAS binary op (for illustration only)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbbinopinfo (binop)
// gbbinopinfo (binop, type)

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

    gb_usage (nargin >= 1 && nargin <= 2 && nargout == 0, USAGE) ;

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

    if (idxunop != NULL)
    {
        OK (GxB_IndexUnaryOp_fprint (idxunop, opstring, GxB_COMPLETE, NULL)) ;
    }
    else
    {
        OK (GxB_BinaryOp_fprint (op2, opstring, GxB_COMPLETE, NULL)) ;
    }
    GB_WRAPUP ;
}

