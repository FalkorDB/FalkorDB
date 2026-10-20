//------------------------------------------------------------------------------
// GB_emult_iso.h
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_EMULT_ISO_H
#define GB_EMULT_ISO_H

bool GB_emult_iso           // c = op(a,b), return true if C is iso
(
    // output
    GB_void *restrict c,    // output scalar of iso array
    // input
    GrB_Type ctype,         // type of c
    GrB_Matrix A,           // input matrix
    GrB_Matrix B,           // input matrix
    GrB_BinaryOp op         // binary operator
) ;

#endif

