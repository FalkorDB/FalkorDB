//------------------------------------------------------------------------------
// gb_string_to_monoid: get a GraphBLAS monoid from a string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

// The string has the form op_name.op_type.  For example '+.double' is
// GrB_PLUS_MONOID_FP64.  The type is optional.  If not present, it defaults
// to the default_type parameter.

GrB_Monoid gb_string_to_monoid          // return monoid from a string
(
    char *opstring,                     // string defining the operator
    const GrB_Type type                 // default type if not in the string
)
{ 

    // get the binary operator and convert to a monoid
    return (gb_binop_to_monoid (gb_string_to_binop_or_idxunop (opstring,
        type, type, NULL, NULL))) ;
}

