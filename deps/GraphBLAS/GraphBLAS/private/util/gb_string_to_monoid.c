//------------------------------------------------------------------------------
// gb_string_to_monoid: get a GraphBLAS monoid from a string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The string has the form op_name.op_type.  For example '+.double' is
// GrB_PLUS_MONOID_FP64.  The type is optional.  If not present, it defaults
// to the default_type parameter.

GrB_Info gb_string_to_monoid            // return monoid from a string
(
    // output
    GrB_Monoid *monoid,
    // input
    char *opstring,                     // string defining the operator
    const GrB_Type type,                // default type if not in the string
    char err [ERRLEN]
)
{ 

    //--------------------------------------------------------------------------
    // get the binary operator defined by the opstring and type
    //--------------------------------------------------------------------------

    ASSERT (monoid != NULL) ;
    GrB_BinaryOp binop = NULL ;
    OK (gb_string_to_binop_or_idxunop (&binop,
        /* idxunop: not allowed here: */ NULL, NULL,
        opstring, type, type, err)) ;

    //--------------------------------------------------------------------------
    // convert the binary op to a monoid and return result
    //--------------------------------------------------------------------------

    OK (gb_binop_to_monoid (monoid, binop, err)) ;
    return (GrB_SUCCESS) ;
}

