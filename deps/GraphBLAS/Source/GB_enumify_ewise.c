//------------------------------------------------------------------------------
// GB_enumify_ewise: enumerate a GrB_eWise* problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Enumify an ewise operation: eWiseAdd, eWiseMult, eWiseUnion,
// rowscale, colscale, apply with bind 1st and 2nd, transpose apply with
// bind 1st and 2nd, etc.

#include "GB.h"
#include "GB_stringify.h"

// accum is not present.  Kernels that use it would require accum to be
// the same as the binary operator (but this may change in the future).

void GB_enumify_ewise       // enumerate a GrB_eWise problem
(
    // output:
    uint64_t *scode,        // unique encoding of the entire operation
    // input:
    bool is_eWiseMult,      // if true, method is emult
    bool is_eWiseUnion,     // if true, method is eWiseUnion
    bool can_copy_to_C,     // if true C(i,j)=A(i,j) can bypass the op
    // C matrix:
    bool C_iso,             // if true, C is iso on output
    bool C_in_iso,          // if true, C is iso on input
    int C_sparsity,         // sparse, hyper, bitmap, or full
    GrB_Type ctype,         // C=((ctype) T) is the final typecast
    // M matrix:
    GrB_Matrix M,           // may be NULL
    bool Mask_struct,       // mask is structural
    bool Mask_comp,         // mask is complemented
    // operator:
    GrB_BinaryOp binaryop,  // the binary operator to enumify
    bool flipxy,            // multiplier is: op(a,b) or op(b,a)
    // A and B:
    GrB_Matrix A,           // NULL for unary apply with binop, bind 1st
    GrB_Matrix B            // NULL for unary apply with binop, bind 2nd
)
{

    //--------------------------------------------------------------------------
    // get the types of A, B, and M
    //--------------------------------------------------------------------------

    GrB_Type atype = (A == NULL) ? NULL : A->type ;
    GrB_Type btype = (B == NULL) ? NULL : B->type ;
    GrB_Type mtype = (M == NULL) ? NULL : M->type ;

    //--------------------------------------------------------------------------
    // get the types of X, Y, and Z, and handle the C_iso case, and GB_wait
    //--------------------------------------------------------------------------

    GB_Opcode binaryop_opcode ;
    GB_Type_code xcode, ycode, zcode ;
    ASSERT (binaryop != NULL) ;

    if (C_iso)
    { 
        // values of C are not computed by the kernel
        binaryop_opcode = GB_PAIR_binop_code ;
        xcode = 0 ;
        ycode = 0 ;
        zcode = 0 ;
    }
    else
    { 
        // normal case
        binaryop_opcode = binaryop->opcode ;
        xcode = binaryop->xtype->code ;
        ycode = binaryop->ytype->code ;
        zcode = binaryop->ztype->code ;
    }

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

    if (xcode == GB_BOOL_code)  // && (ycode == GB_BOOL_code)
    { 
        // rename the operator
        binaryop_opcode = GB_boolean_rename (binaryop_opcode) ;
    }

    //--------------------------------------------------------------------------
    // determine if A and/or B are value-agnostic
    //--------------------------------------------------------------------------

    // These 1st, 2nd, and pair operators are all handled by the flip, so if
    // flipxy is still true, all of these booleans will be false.
    bool op_is_first  = (binaryop_opcode == GB_FIRST_binop_code ) ;
    bool op_is_second = (binaryop_opcode == GB_SECOND_binop_code) ;
    bool op_is_pair   = (binaryop_opcode == GB_PAIR_binop_code) ;
    bool op_is_positional = GB_OPCODE_IS_POSITIONAL (binaryop_opcode) ;

    if (op_is_positional || op_is_pair || C_iso)
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

    if (is_eWiseMult || is_eWiseUnion)
    { 
        A_is_pattern = (xcode == 0) ;   // A is not needed if x is not used
        B_is_pattern = (ycode == 0) ;   // B is not needed if x is not used
    }

    //--------------------------------------------------------------------------
    // enumify the binary operator
    //--------------------------------------------------------------------------

    int binop_ecode ;
    GB_enumify_binop (&binop_ecode, binaryop_opcode, xcode, false) ;

    int is_union  = (is_eWiseUnion) ? 1 : 0 ;
    int is_emult  = (is_eWiseMult ) ? 1 : 0 ;
    int copy_to_C = (can_copy_to_C) ? 1 : 0 ;

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

    //--------------------------------------------------------------------------
    // construct the ewise scode
    //--------------------------------------------------------------------------

    // total scode bits: 51 (13 hex digits)

    (*scode) =
                                               // range        bits
                // method (3 bits) (1 hex digit, 0 to 7)
                GB_LSHIFT (is_emult   , 50) |  // 0 or 1       1
                GB_LSHIFT (is_union   , 49) |  // 0 or 1       1
                GB_LSHIFT (copy_to_C  , 48) |  // 0 or 1       1

                // C in, A and B iso properites, flipxy (1 hex digit)
                GB_LSHIFT (C_in_iso_cd, 47) |  // 0 or 1       1
                GB_LSHIFT (A_iso_code , 46) |  // 0 or 1       1
                GB_LSHIFT (B_iso_code , 45) |  // 0 or 1       1
                GB_LSHIFT (flipxy     , 44) |  // 0 or 1       1

                // binaryop, z = f(x,y) (5 hex digits)
                GB_LSHIFT (binop_ecode, 36) |  // 0 to 254     8
                GB_LSHIFT (zcode      , 32) |  // 0 to 14      4
                GB_LSHIFT (xcode      , 28) |  // 0 to 14      4
                GB_LSHIFT (ycode      , 24) |  // 0 to 14      4

                // mask (one hex digit)
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

