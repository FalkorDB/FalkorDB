//------------------------------------------------------------------------------
// GB_positional.h: positional functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_POSITIONAL_H
#define GB_POSITIONAL_H

GrB_UnaryOp GB_positional_unop_ijflip   // return flipped operator
(
    GrB_UnaryOp op                      // operator to flip
) ;

GrB_BinaryOp GB_positional_binop_ijflip // return flipped operator
(
    GrB_BinaryOp op                     // operator to flip
) ;

GrB_IndexUnaryOp GB_positional_idxunop_ijflip   // return flipped operator
(
    int64_t *ithunk,            // input/output: revised value of thunk
    GrB_IndexUnaryOp op         // operator to flip
) ;

int64_t GB_positional_offset        // return the positional thunk
(
    GB_Opcode opcode,               // opcode of positional operator
    GrB_Scalar Thunk,               // thunk for idxunops, or NULL
    bool *depends_on_j              // if true, the op depends on j
) ;

// for internal use only
GB_GLOBAL GrB_IndexUnaryOp GxB_FLIPDIAGINDEX_INT32, GxB_FLIPDIAGINDEX_INT64,
    GxB_NONZOMBIE ;

#endif

