//------------------------------------------------------------------------------
// gb_string_to_semiring: convert a string to a GraphBLAS semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Only built-in GraphBLAS types and operators are supported.

#include "gb_interface.h"

GrB_Semiring gb_string_to_semiring      // return a semiring from a string
(
    char *semiring_string,              // string defining the semiring
    const GrB_Type atype,               // type of A
    const GrB_Type btype                // type of B
)
{

    //--------------------------------------------------------------------------
    // get the string and parse it
    //--------------------------------------------------------------------------

    int32_t position [2] ;
    gb_find_dot (position, semiring_string) ;

    // semiring string must have at least one dot
    CHECK_ERROR (position [0] < 0,
        "invalid semiring; must have the form 'add.mult' or 'add.mult.type'") ;

    // get the name of the add operator
    char *add_name  = semiring_string ;
    semiring_string [position [0]] = '\0' ;

    // get the name and optional type of the mult operator
    char *mult_name = semiring_string + position [0] + 1 ;
    char *mult_typename = NULL ;
    if (position [1] >= 0)
    { 
        semiring_string [position [1]] = '\0' ;
        mult_typename = semiring_string + position [1] + 1 ;
    }

    //--------------------------------------------------------------------------
    // get the mult operator
    //--------------------------------------------------------------------------

    bool type_not_given = (mult_typename == NULL) ;
    GrB_Type mult_type ;
    if (mult_typename == NULL)
    { 
        mult_type = gb_default_type (atype, btype) ;
    }
    else
    { 
        mult_type = gb_string_to_type (mult_typename) ;
    }

    GrB_BinaryOp mult = gb_string_and_type_to_binop_or_idxunop (mult_name,
        mult_type, type_not_given, NULL, NULL) ;
    CHECK_ERROR (mult == NULL, "invalid semiring (unknown multipy operator)") ;

    //--------------------------------------------------------------------------
    // get the add operator
    //--------------------------------------------------------------------------

    GrB_Type add_type ;
    OK (GxB_BinaryOp_ztype (&add_type, mult)) ;
    GrB_BinaryOp add = gb_string_and_type_to_binop_or_idxunop (add_name,
        add_type, false, NULL, NULL) ;
    CHECK_ERROR (add == NULL, "invalid semiring (unknown add operator)") ;

    //--------------------------------------------------------------------------
    // convert the add and mult operators to a semiring
    //--------------------------------------------------------------------------

    return (gb_semiring (add, mult)) ;
}

