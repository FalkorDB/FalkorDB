//------------------------------------------------------------------------------
// GB_enumify_ewise: enumerate a GrB_eWise* problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Enumify an ewise operation: eWiseAdd, eWiseMult, eWiseUnion, rowscale,
// colscale, apply with bind 1st and 2nd, transpose apply with bind 1st and
// 2nd, etc.

#include "GB.h"
#include "jitifyer/GB_stringify.h"

// accum is not present.  Kernels that use it would require accum to be
// the same as the binary operator (but this may change in the future).

void GB_enumify_ewise       // enumerate a GrB_eWise problem
(
    // output:
    uint64_t *method_code,  // unique encoding of the entire operation
    // input:
    bool is_eWiseMult,      // if true, method is eWiseMult
    bool is_eWiseUnion,     // if true, method is eWiseUnion
    bool is_kronecker,      // if true, method is kronecker
    bool is_eWiseAdd,       // if true, method is eWiseAdd
    // C matrix:
    bool C_iso,             // if true, C is iso on output
    bool C_in_iso,          // if true, C is iso on input
    int C_sparsity,         // sparse, hyper, bitmap, or full
    GrB_Type ctype,         // C=((ctype) T) is the final typecast
    bool Cp_is_32,          // if true, Cp is 32-bit; else 64-bit
    bool Cj_is_32,          // if true, Cj is 32-bit; else 64-bit
    bool Ci_is_32,          // if true, Ci is 32-bit; else 64-bit
    // M matrix:
    GrB_Matrix M,           // may be NULL
    bool Mask_struct,       // mask is structural
    bool Mask_comp,         // mask is complemented
    // operator:
    GrB_BinaryOp binaryop,  // the binary operator to enumify
    bool flipij,            // multiplier is: op(a,b,i,j) or op(a,b,j,i)
    bool flipxy,            // multiplier is: op(a,b,i,j) or op(b,a,j,i)
    // A and B:
    GrB_Matrix A,           // NULL for unary apply with binop, bind 1st
    GrB_Matrix B            // NULL for unary apply with binop, bind 2nd
)
{

    //--------------------------------------------------------------------------
    // get the types of A, B, and M
    //--------------------------------------------------------------------------

    ASSERT_BINARYOP_OK (binaryop, "binaryop to enumify", GB0) ;
    GrB_Type atype = (A == NULL) ? NULL : A->type ;
    GrB_Type btype = (B == NULL) ? NULL : B->type ;
    GrB_Type mtype = (M == NULL) ? NULL : M->type ;

    //--------------------------------------------------------------------------
    // get the types of X, Y, and Z, and handle the C_iso case, and GB_wait
    //--------------------------------------------------------------------------

    GB_Opcode opcode ;
    GB_Type_code xcode, ycode, zcode ;
    ASSERT (binaryop != NULL) ;

    if (C_iso)
    { 
        // values of C are not computed by the kernel
        opcode = GB_PAIR_binop_code ;
        xcode = 0 ;
        ycode = 0 ;
        zcode = 0 ;
    }
    else
    { 
        // normal case
        opcode = binaryop->opcode ;
        xcode = binaryop->xtype->code ;
        ycode = binaryop->ytype->code ;
        zcode = binaryop->ztype->code ;
        if (xcode == GB_BOOL_code)  // && (ycode == GB_BOOL_code)
        { 
            // rename the operator
            opcode = GB_boolean_rename (opcode) ;
        }
    }

    //--------------------------------------------------------------------------
    // determine if A and/or B are value-agnostic
    //--------------------------------------------------------------------------

    // These 1st, 2nd, and pair operators are all handled by the flip, so if
    // flipxy is still true, all of these booleans will be false.
    bool op_is_first  = (opcode == GB_FIRST_binop_code ) ;
    bool op_is_second = (opcode == GB_SECOND_binop_code) ;
    bool op_is_pair   = (opcode == GB_PAIR_binop_code) ;
    bool op_is_builtin_positional =
        GB_IS_BUILTIN_BINOP_CODE_POSITIONAL (opcode) ;

    if (op_is_builtin_positional || op_is_pair || C_iso)
    { 
        // x and y are not used
        xcode = 0 ;
        ycode = 0 ;
    }
    else if (op_is_second)
    { 
        // x is not used
        xcode = 0 ;
    }
    else if (op_is_first)
    { 
        // y is not used
        ycode = 0 ;
    }

    bool A_is_pattern = false ;
    bool B_is_pattern = false ;

    if (is_eWiseMult || is_eWiseUnion || is_kronecker)
    { 
        A_is_pattern = (xcode == 0) ;   // A is not needed if x is not used
        B_is_pattern = (ycode == 0) ;   // B is not needed if y is not used
    }

    //--------------------------------------------------------------------------
    // enumify the binary operator
    //--------------------------------------------------------------------------

    int binop_code = (opcode - GB_USER_binop_code) & 0x3F ;

    //--------------------------------------------------------------------------
    // enumify the types
    //--------------------------------------------------------------------------

    // If A is NULL (for binop bind 1st), acode is 15
    // If B is NULL (for binop bind 2nd), bcode is 15

    int acode = (A == NULL) ? 15 : (A_is_pattern ? 0 : atype->code) ; // 0 to 15
    int bcode = (B == NULL) ? 15 : (B_is_pattern ? 0 : btype->code) ; // 0 to 15

    int ccode = C_iso ? 0 : ctype->code ;          // 0 to 14

    int A_iso_code = (A != NULL && A->iso) ? 1 : 0 ;
    int B_iso_code = (B != NULL && B->iso) ? 1 : 0 ;
    int C_in_iso_cd = (C_in_iso) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // enumify the mask
    //--------------------------------------------------------------------------

    int mtype_code = (mtype == NULL) ? 0 : mtype->code ; // 0 to 14
    int mask_ecode ;
    GB_enumify_mask (&mask_ecode, mtype_code, Mask_struct, Mask_comp) ;

    //--------------------------------------------------------------------------
    // enumify the sparsity structures of C, M, A, and B
    //--------------------------------------------------------------------------

    int M_sparsity = (M == NULL) ? 0 : GB_sparsity (M) ;
    int A_sparsity = (A == NULL) ? 0 : GB_sparsity (A) ;
    int B_sparsity = (B == NULL) ? 0 : GB_sparsity (B) ;

    int csparsity, msparsity, asparsity, bsparsity ;
    GB_enumify_sparsity (&csparsity, C_sparsity) ;
    GB_enumify_sparsity (&msparsity, M_sparsity) ;
    GB_enumify_sparsity (&asparsity, A_sparsity) ;
    GB_enumify_sparsity (&bsparsity, B_sparsity) ;

    int cp_is_32 = (Cp_is_32) ? 1 : 0 ;
    int cj_is_32 = (Cj_is_32) ? 1 : 0 ;
    int ci_is_32 = (Ci_is_32) ? 1 : 0 ;

    int mp_is_32 = (M == NULL) ? 0 : (M->p_is_32) ? 1 : 0 ;
    int mj_is_32 = (M == NULL) ? 0 : (M->j_is_32) ? 1 : 0 ;
    int mi_is_32 = (M == NULL) ? 0 : (M->i_is_32) ? 1 : 0 ;

    int ap_is_32 = (A == NULL) ? 0 : (A->p_is_32) ? 1 : 0 ;
    int aj_is_32 = (A == NULL) ? 0 : (A->j_is_32) ? 1 : 0 ;
    int ai_is_32 = (A == NULL) ? 0 : (A->i_is_32) ? 1 : 0 ;

    int bp_is_32 = (B == NULL) ? 0 : (B->p_is_32) ? 1 : 0 ;
    int bj_is_32 = (B == NULL) ? 0 : (B->j_is_32) ? 1 : 0 ;
    int bi_is_32 = (B == NULL) ? 0 : (B->i_is_32) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // construct the ewise method_code
    //--------------------------------------------------------------------------

    // total method_code bits: 59 (15 hex digits); 5 bits to spare.

    (*method_code) =
                                               // range        bits
                // C, M, A, B: 32/64 (12 bits) (3 hex digits)
                GB_LSHIFT (cp_is_32   , 59) |  // 0 or 1       1
                GB_LSHIFT (cj_is_32   , 58) |  // 0 or 1       1
                GB_LSHIFT (ci_is_32   , 57) |  // 0 or 1       1

                GB_LSHIFT (mp_is_32   , 56) |  // 0 or 1       1
                GB_LSHIFT (mj_is_32   , 55) |  // 0 or 1       1
                GB_LSHIFT (mi_is_32   , 54) |  // 0 or 1       1

                GB_LSHIFT (ap_is_32   , 53) |  // 0 or 1       1
                GB_LSHIFT (aj_is_32   , 52) |  // 0 or 1       1
                GB_LSHIFT (ai_is_32   , 51) |  // 0 or 1       1

                GB_LSHIFT (bp_is_32   , 50) |  // 0 or 1       1
                GB_LSHIFT (bj_is_32   , 49) |  // 0 or 1       1
                GB_LSHIFT (bi_is_32   , 48) |  // 0 or 1       1

                // C in, A and B iso properites (3 bits) (1 hex digit)
                // one bit unused here
                GB_LSHIFT (C_in_iso_cd, 46) |  // 0 or 1       1
                GB_LSHIFT (A_iso_code , 45) |  // 0 or 1       1
                GB_LSHIFT (B_iso_code , 44) |  // 0 or 1       1

                // binaryop, z = f(x,y) (5 hex digits)
                GB_LSHIFT (flipxy     , 43) |  // 0 or 1       1
                GB_LSHIFT (flipij     , 42) |  // 0 or 1       1
                GB_LSHIFT (binop_code , 36) |  // 0 to 52      6
                GB_LSHIFT (zcode      , 32) |  // 0 to 14      4
                GB_LSHIFT (xcode      , 28) |  // 0 to 14      4
                GB_LSHIFT (ycode      , 24) |  // 0 to 14      4

                // mask (1 hex digit)
                GB_LSHIFT (mask_ecode , 20) |  // 0 to 13      4

                // types of C, A, and B (3 hex digits)
                GB_LSHIFT (ccode      , 16) |  // 0 to 14      4
                GB_LSHIFT (acode      , 12) |  // 0 to 15      4
                GB_LSHIFT (bcode      ,  8) |  // 0 to 15      4

                // sparsity structures of C, M, A, and B (2 hex digits)
                GB_LSHIFT (csparsity  ,  6) |  // 0 to 3       2
                GB_LSHIFT (msparsity  ,  4) |  // 0 to 3       2
                GB_LSHIFT (asparsity  ,  2) |  // 0 to 3       2
                GB_LSHIFT (bsparsity  ,  0) ;  // 0 to 3       2
}

