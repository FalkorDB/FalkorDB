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
    const GrB_BinaryOp dup,     // operator for duplicates
    const GrB_Type ttype,       // type of Tx
    const GrB_Type stype,       // type of Sx array (values of input tuples)
    bool is_matrix,             // if true, J is NULL, else non-NULL
    bool iso_build,             // if true, Tx and Sx are iso
    bool Tp_is_32,              // if true, Tp is uint32_t, else uint64_t
    bool Tj_is_32,              // if true, Tj is uint32_t, else uint64_t
    bool Ti_is_32,              // if true, Ti is uint32_t, else uint64_t
    bool I_is_32,               // if true, I is uint32_t else uint64_t
    bool J_is_32,               // if true, J is uint32_t else uint64_t
    bool K_is_32,               // if true, K_work is uint32_t else uint64_t
    bool K_is_null,             // if true, K_work is NULL
    bool Key_preloaded,         // if true, Key_in is preloaded on input
    bool Key_is_32,             // if true, GB_key_t is uint32_t else uint64_t
    bool known_no_duplicates,   // if true, tuples known to not have duplicates
    bool known_sorted           // if true, tuples known to already be sorted
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

    int iso       = (iso_build) ? 1 : 0 ;
    int is_mat    = (is_matrix) ? 1 : 0 ;
    int tp_is_32  = (Tp_is_32)  ? 1 : 0 ;
    int tj_is_32  = (Tj_is_32)  ? 1 : 0 ;
    int ti_is_32  = (Ti_is_32)  ? 1 : 0 ;
    int i_is_32   = (I_is_32)   ? 1 : 0 ;
    int j_is_32   = (J_is_32)   ? 1 : 0 ;
    int k_is_32   = (K_is_32)   ? 1 : 0 ;
    int k_is_null = (K_is_null) ? 1 : 0 ;
    int key_pre   = (Key_preloaded) ? 1 : 0 ;
    int key_is_32 = (Key_is_32) ? 1 : 0 ;
    int no_dupl   = (known_no_duplicates) ? 1 : 0 ;
    int sorted    = (known_sorted) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // enumify the dup binary operator
    //--------------------------------------------------------------------------

    int dup_code = (dup_opcode - GB_USER_binop_code) & 0x3F ;

    //--------------------------------------------------------------------------
    // construct the method_code
    //--------------------------------------------------------------------------

    // total method_code bits: 39 (10 hex digits)

    (*method_code) =
                                               // range        bits
                // Key_in preloaded, and size of integers in GB_Key_t:
                GB_LSHIFT (sorted     , 38) |  // 0 to 1       1
                GB_LSHIFT (key_pre    , 37) |  // 0 to 1       1
                GB_LSHIFT (key_is_32  , 36) |  // 0 to 1       1

                // 32/64 bit (2 hex digits)
                GB_LSHIFT (is_mat     , 35) |  // 0 to 1       1
                GB_LSHIFT (j_is_32    , 34) |  // 0 to 1       1
                GB_LSHIFT (tp_is_32   , 33) |  // 0 to 1       1
                GB_LSHIFT (tj_is_32   , 32) |  // 0 to 1       1
                GB_LSHIFT (ti_is_32   , 31) |  // 0 to 1       1
                GB_LSHIFT (i_is_32    , 30) |  // 0 to 1       1
                GB_LSHIFT (k_is_32    , 29) |  // 0 to 1       1
                GB_LSHIFT (k_is_null  , 28) |  // 0 to 1       1

                // dup, z = f(x,y), and iso flag (5 hex digits)
                GB_LSHIFT (no_dupl    , 27) |  // 0 to 1       1
                GB_LSHIFT (iso        , 26) |  // 0 to 1       1
                GB_LSHIFT (dup_code   , 20) |  // 0 to 52      6
                GB_LSHIFT (zcode      , 16) |  // 0 to 14      4
                GB_LSHIFT (xcode      , 12) |  // 0 to 14      4
                GB_LSHIFT (ycode      ,  8) |  // 0 to 14      4

                // types of S and T (2 hex digits)
                GB_LSHIFT (tcode      ,  4) |  // 0 to 14      4
                GB_LSHIFT (scode      ,  0) ;  // 0 to 15      4
}

