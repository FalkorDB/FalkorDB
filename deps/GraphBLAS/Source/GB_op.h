//------------------------------------------------------------------------------
// GB_op.h: definitions for operators
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_OPERATOR_H
#define GB_OPERATOR_H

GrB_Info GB_Op_free             // free a user-created op
(
    GB_Operator *op_handle      // handle of operator to free
) ;

bool GB_op_is_second    // return true if op is SECOND, of the right type
(
    GrB_BinaryOp op,
    GrB_Type type
) ;

GrB_Info GB_op_name_and_defn
(
    // output
    char *op_name,              // op->name of the GrB operator struct
    int32_t *op_name_len,       // op->name_len
    uint64_t *op_hash,          // op->hash
    char **op_defn,             // op->defn
    size_t *op_defn_size,       // op->defn_size
    // input
    const char *input_name,     // user-provided name, may be NULL
    const char *input_defn,     // user-provided name, may be NULL
    bool user_op,               // if true, a user-defined op
    bool jitable                // if true, the op can be JIT'd
) ;

#endif

