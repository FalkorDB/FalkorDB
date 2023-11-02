//------------------------------------------------------------------------------
// gb_string_to_unop: get a GraphBLAS unary operator from a string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

// The string has the form op_name.op_type.  For example 'abs.double' is the
// GrB_ABS_FP64 operator.  The type is optional.  If not present, it defaults
// to the default_type parameter.

GrB_UnaryOp gb_string_to_unop           // return unary operator from a string
(
    char *opstring,                     // string defining the operator
    const GrB_Type default_type         // default type if not in the string
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
    if (op_typename == NULL)
    { 
        type = default_type ;
    }
    else
    { 
        type = gb_string_to_type (op_typename) ;
    }

    //--------------------------------------------------------------------------
    // convert the string to a GraphBLAS unary operator, built-in or Complex
    //--------------------------------------------------------------------------

    return (gb_string_and_type_to_unop (op_name, type, type_not_given)) ;
}

