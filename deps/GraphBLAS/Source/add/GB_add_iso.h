//------------------------------------------------------------------------------
// GB_add_iso.h: definitions for GB_add and related functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_ADD_ISO_H
#define GB_ADD_ISO_H

bool GB_add_iso             // c = op(a,b), return true if C is iso
(
    // output
    GB_void *restrict c,    // output scalar of iso array
    // input
    GrB_Type ctype,         // type of c
    GrB_Matrix A,           // input matrix
    const GB_void *restrict alpha_scalar,   // of type op->xtype
    GrB_Matrix B,           // input matrix
    const GB_void *restrict beta_scalar,    // of type op->ytype
    GrB_BinaryOp op,        // binary operator
    const bool A_and_B_are_disjoint,
    const bool is_eWiseUnion
) ;

#endif

