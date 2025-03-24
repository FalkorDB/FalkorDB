//------------------------------------------------------------------------------
// GB_enumify_select: enumerate a GrB_select problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

// Currently, the mask M and the accum are not present, and C and A have the
// same type, but these conditions may change in the future.

void GB_enumify_select      // enumerate a GrB_selectproblem
(
    // output:
    uint64_t *method_code,  // unique encoding of the entire operation
    // input:
    const GrB_Matrix C,
    const GrB_IndexUnaryOp op,   // the index unary operator to enumify
    const bool flipij,           // if true, flip i and j
    const GrB_Matrix A
)
{

    //--------------------------------------------------------------------------
    // get the types of A, X, Y, and Z
    //--------------------------------------------------------------------------

    GrB_Type atype = A->type ;
    GB_Opcode opcode = op->opcode ;
    GB_Type_code zcode = op->ztype->code ;
    GB_Type_code xcode = ((op->xtype == NULL) ? 0 : op->xtype->code) ;
    GB_Type_code ycode = op->ytype->code ;
    ASSERT (A->type == C->type) ;

    //--------------------------------------------------------------------------
    // enumify the idxunop operator
    //--------------------------------------------------------------------------

    ASSERT (opcode >= GB_ROWINDEX_idxunop_code) ;
    ASSERT (opcode <= GB_USER_idxunop_code) ;
    int idxop_code = opcode - GB_ROWINDEX_idxunop_code ;

    //--------------------------------------------------------------------------
    // enumify the types
    //--------------------------------------------------------------------------

    int acode = atype->code ;               // 1 to 14
    int A_iso_code = (A->iso) ? 1 : 0 ;
    int C_iso_code = (C->iso) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // enumify the sparsity structure of A and C
    //--------------------------------------------------------------------------

    int asparsity, csparsity ;
    GB_enumify_sparsity (&csparsity, GB_sparsity (C)) ;
    GB_enumify_sparsity (&asparsity, GB_sparsity (A)) ;

    int cp_is_32 = (C->p_is_32) ? 1 : 0 ;
    int cj_is_32 = (C->j_is_32) ? 1 : 0 ;
    int ci_is_32 = (C->i_is_32) ? 1 : 0 ;

    int ap_is_32 = (A->p_is_32) ? 1 : 0 ;
    int aj_is_32 = (A->j_is_32) ? 1 : 0 ;
    int ai_is_32 = (A->i_is_32) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // construct the select method_code
    //--------------------------------------------------------------------------

    // total method_code bits:  34 (9 hex digits)

    (*method_code) =
                                               // range        bits
                // C, A: 32/64 (2 hex digits)
                GB_LSHIFT (cp_is_32   , 33) |  // 0 or 1       1
                GB_LSHIFT (cj_is_32   , 32) |  // 0 or 1       1
                GB_LSHIFT (ci_is_32   , 31) |  // 0 or 1       1

                GB_LSHIFT (ap_is_32   , 30) |  // 0 or 1       1
                GB_LSHIFT (aj_is_32   , 29) |  // 0 or 1       1
                GB_LSHIFT (ai_is_32   , 28) |  // 0 or 1       1

                // op, z = f(x,i,j,y), flipij, and iso codes (5 hex digits)
                GB_LSHIFT (C_iso_code , 27) |  // 0 or 1       1
                GB_LSHIFT (A_iso_code , 26) |  // 0 or 1       1
                GB_LSHIFT (flipij     , 25) |  // 0 or 1       1
                GB_LSHIFT (idxop_code , 20) |  // 0 to 19      5
                GB_LSHIFT (zcode      , 16) |  // 0 to 14      4
                GB_LSHIFT (xcode      , 12) |  // 0 to 14      4
                GB_LSHIFT (ycode      ,  8) |  // 0 to 14      4

                // type of (1 hex digit)
                GB_LSHIFT (acode      ,  4) |  // 0 to 15      4

                // sparsity structures of C and A (1 hex digit)
                GB_LSHIFT (csparsity  ,  2) |  // 0 to 3       2
                GB_LSHIFT (asparsity  ,  0) ;  // 0 to 3       2
}

