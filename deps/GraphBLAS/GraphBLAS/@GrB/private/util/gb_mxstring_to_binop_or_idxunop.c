//------------------------------------------------------------------------------
// gb_mxstring_to_binop_or_idxunop: get a GraphBLAS op from a built-in string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

void gb_mxstring_to_binop_or_idxunop    // binop or idxunop from a string
(
    const mxArray *mxstring,            // built-in string
    const GrB_Type atype,               // type of A
    const GrB_Type btype,               // type of B
    // output:
    GrB_BinaryOp *op2,                  // binary op
    GrB_IndexUnaryOp *idxunop,          // idxunop
    int64_t *ithunk                     // thunk for idxunop
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    (*op2) = NULL ;
    (*idxunop) = NULL ;

    if (gb_mxarray_is_empty (mxstring))
    { 
        // no operator is present, or present and empty
        return ;
    }

    //--------------------------------------------------------------------------
    // get the string
    //--------------------------------------------------------------------------

    #define LEN 256
    char opstring [LEN+2] ;
    gb_mxstring_to_string (opstring, LEN, mxstring, "binary/index operator") ;

    //--------------------------------------------------------------------------
    // convert the string to a binary operator or index unary operator
    //--------------------------------------------------------------------------

    (*op2) = gb_string_to_binop_or_idxunop (opstring, atype, btype, idxunop,
        ithunk) ;
}

