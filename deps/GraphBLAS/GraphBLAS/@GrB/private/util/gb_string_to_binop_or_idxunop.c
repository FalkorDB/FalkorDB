//------------------------------------------------------------------------------
// gb_string_to_binop_or_idxunop: get a GraphBLAS operator from a string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

// The string has the form op_name.op_type.  For example '+.double' is the
// GrB_PLUS_FP64 operator.  The type is optional.  If not present in the
// string, it is found by gb_default_type (atype, btype).

GrB_BinaryOp gb_string_to_binop_or_idxunop
(
    char *opstring,                     // string defining the operator
    const GrB_Type atype,               // type of A
    const GrB_Type btype,               // type of B
    GrB_IndexUnaryOp *idxunop,          // idxunop from the string
    int64_t *ithunk                     // thunk for idxunop
)
{

    //--------------------------------------------------------------------------
    // get the string and parse it
    //--------------------------------------------------------------------------

    int32_t position [2] ;
    gb_find_dot (position, opstring) ;

    char *op_name = opstring ;
    char *op_typename = NULL ;
    if (position [0] >= 0)
    { 
        opstring [position [0]] = '\0' ;
        op_typename = opstring + position [0] + 1 ;
    }

    //--------------------------------------------------------------------------
    // get the operator type
    //--------------------------------------------------------------------------

    bool type_not_given = (op_typename == NULL) ;
    GrB_Type type ;
    if (type_not_given)
    { 
        type = gb_default_type (atype, btype) ;
    }
    else
    { 
        type = gb_string_to_type (op_typename) ;
    }

    //--------------------------------------------------------------------------
    // convert the string to a GraphBLAS binary operator, built-in or Complex
    //--------------------------------------------------------------------------

    return (gb_string_and_type_to_binop_or_idxunop (op_name, type,
        type_not_given, idxunop, ithunk)) ;
}

