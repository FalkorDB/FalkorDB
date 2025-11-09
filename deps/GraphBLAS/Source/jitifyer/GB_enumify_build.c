//------------------------------------------------------------------------------
// GB_enumify_build: enumerate a GB_build problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Enumify a build operation.

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_enumify_build           // enumerate a GB_build problem
(
    // output:
    uint64_t *method_code,      // unique encoding of the entire operation
    // input:
    GrB_BinaryOp dup,           // operator for duplicates
    GrB_Type ttype,             // type of Tx
    GrB_Type stype,             // type of Sx
    bool Ti_is_32,              // if true, Ti is uint32_t, else uint64_t
    bool I_is_32,               // if true, I_work is uint32_t else uint64_t
    bool K_is_32,               // if true, K_work is uint32_t else uint64_t
    bool K_is_null,             // if true, K_work is NULL
    bool no_duplicates          // if true, no duplicates appear
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

    int ti_is_32  = (Ti_is_32)  ? 1 : 0 ;
    int i_is_32   = (I_is_32)   ? 1 : 0 ;
    int k_is_32   = (K_is_32)   ? 1 : 0 ;
    int k_is_null = (K_is_null) ? 1 : 0 ;
    int no_dupl   = (no_duplicates) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // enumify the dup binary operator
    //--------------------------------------------------------------------------

    int dup_code = (dup_opcode - GB_USER_binop_code) & 0x3F ;

    //--------------------------------------------------------------------------
    // construct the method_code
    //--------------------------------------------------------------------------

    // total method_code bits: 31 (8 hex digits)

    (*method_code) =
                                               // range        bits
                // 32/64 bit (1 hex digit)
                GB_LSHIFT (ti_is_32   , 31) |  // 0 to 1       1
                GB_LSHIFT (i_is_32    , 30) |  // 0 to 1       1
                GB_LSHIFT (k_is_32    , 29) |  // 0 to 1       1
                GB_LSHIFT (k_is_null  , 28) |  // 0 to 1       1

                // dup, z = f(x,y) (6 hex digits)
                GB_LSHIFT (no_dupl    , 27) |  // 0 to 1       1
                // 1 bit unused here
                GB_LSHIFT (dup_code   , 20) |  // 0 to 52      6
                GB_LSHIFT (zcode      , 16) |  // 0 to 14      4
                GB_LSHIFT (xcode      , 12) |  // 0 to 14      4
                GB_LSHIFT (ycode      ,  8) |  // 0 to 14      4

                // types of S and T (2 hex digits)
                GB_LSHIFT (tcode      ,  4) |  // 0 to 14      4
                GB_LSHIFT (scode      ,  0) ;  // 0 to 15      4
}

