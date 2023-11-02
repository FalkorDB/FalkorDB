//------------------------------------------------------------------------------
// GB_enumify_build: enumerate a GB_build problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Enumify an build operation.

#include "GB.h"
#include "GB_stringify.h"

void GB_enumify_build       // enumerate a GB_build problem
(
    // output:
    uint64_t *build_code,   // unique encoding of the entire operation
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

    //--------------------------------------------------------------------------
    // rename redundant boolean operators
    //--------------------------------------------------------------------------

    // consider z = op(x,y) where both x and y are boolean:
    // DIV becomes FIRST
    // RDIV becomes SECOND
    // MIN and TIMES become LAND
    // MAX and PLUS become LOR
    // NE, ISNE, RMINUS, and MINUS become LXOR
    // ISEQ becomes EQ
    // ISGT becomes GT
    // ISLT becomes LT
    // ISGE becomes GE
    // ISLE becomes LE

    if (xcode == GB_BOOL_code)
    { 
        // rename the operator
        dup_opcode = GB_boolean_rename (dup_opcode) ;
    }

    //--------------------------------------------------------------------------
    // enumify the dup binary operator
    //--------------------------------------------------------------------------

    int dup_ecode ;
    GB_enumify_binop (&dup_ecode, dup_opcode, xcode, false) ;

    //--------------------------------------------------------------------------
    // construct the build_code
    //--------------------------------------------------------------------------

    // total build_code bits:  28 (7 hex digits)

    (*build_code) =
                                               // range        bits
                // dup, z = f(x,y) (5 hex digits)
                GB_LSHIFT (dup_ecode  , 20) |  // 0 to 254     8
                GB_LSHIFT (zcode      , 16) |  // 0 to 14      4
                GB_LSHIFT (xcode      , 12) |  // 0 to 14      4
                GB_LSHIFT (ycode      ,  8) |  // 0 to 14      4

                // types of S and T (2 hex digits)
                GB_LSHIFT (tcode      ,  4) |  // 0 to 14      4
                GB_LSHIFT (scode      ,  0) ;  // 0 to 15      4
}

