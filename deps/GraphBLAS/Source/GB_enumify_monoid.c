//------------------------------------------------------------------------------
// GB_enumify_monoid: enumify a monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_enumify_monoid  // enumerate a monoid
(
    // outputs:
    int *add_ecode,     // binary op as an enum
    int *id_ecode,      // identity value as an enum
    int *term_ecode,    // terminal value as an enum
    // inputs:
    int add_opcode,     // binary operator of the monoid
    int zcode           // type of the monoid (x, y, and z)
)
{ 

    GB_enumify_binop (add_ecode, add_opcode, zcode, false) ;
    ASSERT (*add_ecode < 32) ;
    GB_enumify_identity (id_ecode, add_opcode, zcode) ;
    GB_enumify_terminal (term_ecode, add_opcode, zcode) ;
}

