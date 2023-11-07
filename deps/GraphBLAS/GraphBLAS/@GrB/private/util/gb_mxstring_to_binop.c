//------------------------------------------------------------------------------
// gb_mxstring_to_binop: get a GraphBLAS operator from a built-in string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

GrB_BinaryOp gb_mxstring_to_binop       // return binary operator from a string
(
    const mxArray *mxstring,            // built-in string
    const GrB_Type atype,               // type of A
    const GrB_Type btype                // type of B
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (gb_mxarray_is_empty (mxstring))
    { 
        // no operator is present, or present and empty; this is not yet an
        // error, since many uses of GraphBLAS functions use an optional accum
        // operator.
        return (NULL) ;
    }

    //--------------------------------------------------------------------------
    // get the string
    //--------------------------------------------------------------------------

    #define LEN 256
    char opstring [LEN+2] ;
    gb_mxstring_to_string (opstring, LEN, mxstring, "binary operator") ;

    //--------------------------------------------------------------------------
    // convert the string to a binary operator
    //--------------------------------------------------------------------------

    return (gb_string_to_binop_or_idxunop (opstring, atype, btype, NULL, NULL));
}

