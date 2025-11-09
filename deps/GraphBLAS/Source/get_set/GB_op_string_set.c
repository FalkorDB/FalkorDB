//------------------------------------------------------------------------------
// GB_op_string_set: set the name or defn of an operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

GrB_Info GB_op_string_set
(
    GB_Operator op,
    char * value,
    int field
)
{ 

    GB_Opcode opcode = op->opcode ;
    bool user_defined = (opcode == GB_USER_unop_code) ||
        (opcode == GB_USER_idxunop_code) ||
        (opcode == GB_USER_binop_code) ||
        (opcode == GB_USER_idxbinop_code) ;

    bool jitable =
        (op->ztype->hash != UINT64_MAX) &&
        (op->xtype->hash != UINT64_MAX) &&
        (op->ytype == NULL || op->ytype->hash != UINT64_MAX) &&
        (op->theta_type == NULL || op->theta_type->hash != UINT64_MAX) ;

    return (GB_op_or_type_string_set (user_defined, jitable, value, field,
        &(op->user_name), &(op->user_name_size),
        op->name, &(op->name_len), &(op->defn), &(op->defn_size),
        &(op->hash))) ;
}

