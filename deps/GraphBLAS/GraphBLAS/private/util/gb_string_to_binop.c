//------------------------------------------------------------------------------
// gb_string_to_binop: get a GraphBLAS operator from a string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GrB_Info gb_string_to_binop // return binary operator from a string
(
    // output
    GrB_BinaryOp *binop,        // binary op determined from the string
    // input/output:
    char *opstring,             // string that defines the binary operator
    // input:
    const GrB_Type atype,       // type of A
    const GrB_Type btype,       // type of B
    char err [ERRLEN]
)
{ 

    // convert the string to a binary operator
    return (gb_string_to_binop_or_idxunop (binop,
        /* idxunop not allowed here: */ NULL, NULL,
        opstring, atype, btype, err)) ;
}

