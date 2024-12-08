//------------------------------------------------------------------------------
// GB_enumify_build: enumerate a GB_build problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Enumify a build operation.

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_enumify_build       // enumerate a GB_build problem
(
    // output:
    uint64_t *method_code,  // unique encoding of the entire operation
    // input:
    GrB_BinaryOp dup,       // operator for duplicates
    GrB_Type ttype,         // type of Tx
    GrB_Type stype          // type of Sx
)
{ 

    //--------------------------------------------------------------------------
    // get the types of X, Y, Z, S, and T
    //--------------------------------------------------------------------------

    ASSERT (dup != NULL) ;
    GB_Opcode dup_opcode = dup->opcode ;
    GB_Type_code xcode = dup->xtype->code ;
    GB_Type_code ycode = dup->ytype->code ;
    GB_Type_code zcode = dup->ztype->code ;
    GB_Type_code tcode = ttype->code ;
    GB_Type_code scode = stype->code ;
    if (xcode == GB_BOOL_code)
    { 
        // rename the operator
        dup_opcode = GB_boolean_rename (dup_opcode) ;
    }

    //--------------------------------------------------------------------------
    // enumify the dup binary operator
    //--------------------------------------------------------------------------

    int dup_code = (dup_opcode - GB_USER_binop_code) & 0x3F ;

    //--------------------------------------------------------------------------
    // construct the method_code
    //--------------------------------------------------------------------------

    // total method_code bits: 26 (7 hex digits)

    (*method_code) =
                                               // range        bits
                // dup, z = f(x,y) (5 hex digits)
                GB_LSHIFT (dup_code   , 20) |  // 0 to 52      6
                GB_LSHIFT (zcode      , 16) |  // 0 to 14      4
                GB_LSHIFT (xcode      , 12) |  // 0 to 14      4
                GB_LSHIFT (ycode      ,  8) |  // 0 to 14      4

                // types of S and T (2 hex digits)
                GB_LSHIFT (tcode      ,  4) |  // 0 to 14      4
                GB_LSHIFT (scode      ,  0) ;  // 0 to 15      4
}

