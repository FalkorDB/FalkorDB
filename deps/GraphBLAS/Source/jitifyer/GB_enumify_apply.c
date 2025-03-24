//------------------------------------------------------------------------------
// GB_enumify_apply: enumerate a GrB_apply problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Enumify an apply or transpose/apply operation.  No accum or mask.  The iso
// cases for C is not handled.  The op is either unary or index unary, not
// binary (that is handled as an ewise enumify).

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_enumify_apply       // enumerate an apply or tranpose/apply problem
(
    // output:
    uint64_t *method_code,  // unique encoding of the entire operation
    // input:
    // C matrix:
    const int C_sparsity,   // sparse, hyper, bitmap, or full.  For apply
                            // without transpose, Cx = op(A) is computed where
                            // Cx is just C->x, so the caller uses 'full' when
                            // C is sparse, hyper, or full.
    const bool C_is_matrix, // true for C=op(A), false for Cx=op(A)
    const GrB_Type ctype,   // C=((ctype) T) is the final typecast
    const bool Cp_is_32,        // if true, Cp is uint32_t, else uint64_t
    const bool Ci_is_32,        // if true, Ci is uint32_t, else uint64_t
    const bool Cj_is_32,        // if true, Cj is uint32_t, else uint64_t
    // operator:
        const GB_Operator op,       // unary/index-unary to apply; not binaryop
        const bool flipij,          // if true, flip i,j for user idxunop
    // A matrix:
    const int A_sparsity,
    const bool A_is_matrix,
    const GrB_Type atype,
    const bool Ap_is_32,        // if true, A->p is uint32_t, else uint64_t
    const bool Aj_is_32,        // if true, A->h is uint32_t, else uint64_t
    const bool Ai_is_32,        // if true, A->i is uint32_t, else uint64_t
    const bool A_iso,
    const int64_t A_nzombies
)
{ 

    //--------------------------------------------------------------------------
    // get the types of X, Y, and Z
    //--------------------------------------------------------------------------

    ASSERT (op != NULL) ;
    GB_Opcode opcode = op->opcode ;
    GrB_Type ztype = op->ztype ;
    GrB_Type xtype = op->xtype ;
    GrB_Type ytype = op->ytype ;
    GB_Type_code zcode = (ztype == NULL) ? 0 : ztype->code ;
    GB_Type_code xcode = (xtype == NULL) ? 0 : xtype->code ;
    GB_Type_code ycode = (ytype == NULL) ? 0 : ytype->code ;

    //--------------------------------------------------------------------------
    // enumify the unary operator
    //--------------------------------------------------------------------------

    bool depends_on_x, depends_on_i, depends_on_j, depends_on_y ;
    int unop_ecode ;
    GB_enumify_unop (&unop_ecode, &depends_on_x, &depends_on_i, &depends_on_j,
        &depends_on_y, flipij, opcode, xcode) ;

    if (!depends_on_x)
    { 
        xcode = 0 ;
    }

    if (!depends_on_y)
    { 
        ycode = 0 ;
    }

    int i_dep = (depends_on_i) ? 1 : 0 ;
    int j_dep = (depends_on_j) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // enumify the types of C and A
    //--------------------------------------------------------------------------

    int acode = (xcode == 0) ? 0 : (atype->code) ;          // 0 to 14
    int ccode = ctype->code ;                               // 0 to 14

    //--------------------------------------------------------------------------
    // enumify the sparsity structures of C and A
    //--------------------------------------------------------------------------

    int csparsity, asparsity ;
    GB_enumify_sparsity (&csparsity, C_sparsity) ;
    GB_enumify_sparsity (&asparsity, A_sparsity) ; // was GB_sparsity (A) ;
    int C_mat = (C_is_matrix) ? 1 : 0 ;
    int A_mat = (A_is_matrix) ? 1 : 0 ;
    int A_iso_code = (A_iso) ? 1 : 0 ;
    int A_zombies = (A_nzombies > 0) ? 1 : 0 ;
    int cp_is_32 = (Cp_is_32) ? 1 : 0 ;
    int ci_is_32 = (Ci_is_32) ? 1 : 0 ;
    int cj_is_32 = (Cj_is_32) ? 1 : 0 ;
    int ap_is_32 = (Ap_is_32) ? 1 : 0 ;
    int aj_is_32 = (Aj_is_32) ? 1 : 0 ;
    int ai_is_32 = (Ai_is_32) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // construct the apply method_code
    //--------------------------------------------------------------------------

    // total method_code bits: 45 bits (12 hex digits)

    (*method_code) =
                                               // range        bits
                // C and A properties (3 hex digits)
                GB_LSHIFT (cp_is_32   , 44) |  // 0 or 1       1
                GB_LSHIFT (ci_is_32   , 43) |  // 0 or 1       1
                GB_LSHIFT (cj_is_32   , 42) |  // 0 or 1       1
                GB_LSHIFT (ap_is_32   , 41) |  // 0 or 1       1
                GB_LSHIFT (aj_is_32   , 40) |  // 0 or 1       1
                GB_LSHIFT (ai_is_32   , 39) |  // 0 or 1       1
                GB_LSHIFT (A_mat      , 38) |  // 0 or 1       1
                GB_LSHIFT (A_zombies  , 37) |  // 0 or 1       1
                GB_LSHIFT (A_iso_code , 36) |  // 0 or 1       1

                // C kind, i/j dependency and flipij (1 hex digit)
                GB_LSHIFT (C_mat      , 35) |  // 0 or 1       1
                GB_LSHIFT (i_dep      , 34) |  // 0 or 1       1
                GB_LSHIFT (j_dep      , 33) |  // 0 or 1       1
                GB_LSHIFT (flipij     , 32) |  // 0 or 1       1

                // op, z = f(x,i,j,y) (5 hex digits)
                GB_LSHIFT (unop_ecode , 24) |  // 0 to 254     8
                GB_LSHIFT (zcode      , 20) |  // 0 to 14      4
                GB_LSHIFT (xcode      , 16) |  // 0 to 14      4
                GB_LSHIFT (ycode      , 12) |  // 0 to 14      4

                // types of C and A (2 hex digits)
                GB_LSHIFT (ccode      ,  8) |  // 0 to 14      4
                GB_LSHIFT (acode      ,  4) |  // 0 to 14      4

                // sparsity structures of C and A (1 hex digit)
                GB_LSHIFT (csparsity  ,  2) |  // 0 to 3       2
                GB_LSHIFT (asparsity  ,  0) ;  // 0 to 3       2
}

