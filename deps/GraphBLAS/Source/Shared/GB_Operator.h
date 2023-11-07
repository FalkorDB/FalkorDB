//------------------------------------------------------------------------------
// GB_Operator.h: definitions of all operator objects
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GrB_UnaryOp, GrB_IndexUnaryOp, GrB_BinaryOp, and GxB_SelectOp all use the
// same internal structure.

    int64_t magic ;         // for detecting uninitialized objects
    size_t header_size ;    // size of the malloc'd block for this struct, or 0
    // ---------------------//
    char *user_name ;       // user name for GrB_get/GrB_set
    size_t user_name_size ; // allocated size of user_name for GrB_get/GrB_set
    // ---------------------//

    GrB_Type ztype ;        // type of z
    GrB_Type xtype ;        // type of x
    GrB_Type ytype ;        // type of y for binop and IndexUnaryOp,
                            // NULL for unaryop

    // function pointers:
    GxB_unary_function       unop_function ;
    GxB_index_unary_function idxunop_function ;
    GxB_binary_function      binop_function ;

    char name [GxB_MAX_NAME_LEN] ;      // JIT C name of the operator
    int32_t name_len ;      // length of JIT C name; 0 for builtin
    GB_Opcode opcode ;      // operator opcode
    char *defn ;            // function definition
    size_t defn_size ;      // allocated size of the definition

    uint64_t hash ;         // if 0, operator uses only builtin ops and types

