//------------------------------------------------------------------------------
// GB_enumify_mxm: enumerate a GrB_mxm problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

// dot3:  C<M>=A'*B, no accum
// saxpy
// inplace C_in is full/bitmap
//      C_in <M> += A*B     monoid ztype doesn't cast (= accum->ytype)
//      C_in <M>  = A*B     monoid ztype casts to C_in->type
// ...

// accum is not present.  Kernels that use it would require accum to be
// the same as the monoid binary operator.

void GB_enumify_mxm         // enumerate a GrB_mxm problem
(
    // output:              // future:: may need to become 2 x uint64
    uint64_t *method_code,  // unique encoding of the entire semiring
    // input:
    // C matrix:
    bool C_iso,             // C output iso: if true, semiring is ANY_PAIR_BOOL
    bool C_in_iso,          // C input iso status
    int C_sparsity,         // sparse, hyper, bitmap, or full
    GrB_Type ctype,         // C=((ctype) T) is the final typecast
    // M matrix:
    GrB_Matrix M,           // may be NULL
    bool Mask_struct,       // mask is structural
    bool Mask_comp,         // mask is complemented
    // semiring:
    GrB_Semiring semiring,  // the semiring to enumify
    bool flipxy,            // multiplier is: mult(a,b) or mult(b,a)
    // A and B:
    GrB_Matrix A,
    GrB_Matrix B
)
{ 

    //--------------------------------------------------------------------------
    // get the semiring
    //--------------------------------------------------------------------------

    ASSERT_SEMIRING_OK (semiring, "semiring for enumify_mxm", GB0) ;
    GrB_Monoid add = semiring->add ;
    GrB_BinaryOp mult = semiring->multiply ;
    GrB_BinaryOp addop = add->op ;

    //--------------------------------------------------------------------------
    // get the types
    //--------------------------------------------------------------------------

    GrB_Type atype = A->type ;
    GrB_Type btype = B->type ;
    GrB_Type mtype = (M == NULL) ? NULL : M->type ;

    GrB_Type xtype = mult->xtype ;
    GrB_Type ytype = mult->ytype ;
    GrB_Type ztype = mult->ztype ;

    GB_Opcode mult_opcode = mult->opcode ;
    GB_Opcode add_opcode  = addop->opcode ;

    GB_Type_code xcode = xtype->code ;
    GB_Type_code ycode = ytype->code ;
    GB_Type_code zcode = ztype->code ;

    // these must always be true for any semiring:
    ASSERT (mult->ztype == addop->ztype) ;
    ASSERT (addop->xtype == addop->ztype && addop->ytype == addop->ztype) ;

    //--------------------------------------------------------------------------
    // rename redundant boolean operators
    //--------------------------------------------------------------------------

    if (C_iso)
    { 
        add_opcode = GB_ANY_binop_code ;
        mult_opcode = GB_PAIR_binop_code ;
        zcode = 0 ;
    }
    else if (zcode == GB_BOOL_code)
    { 
        // rename the monoid
        add_opcode = GB_boolean_rename (add_opcode) ;
    }

    if (xcode == GB_BOOL_code)  // && (ycode == GB_BOOL_code)
    { 
        // rename the multiplicative operator
        mult_opcode = GB_boolean_rename (mult_opcode) ;
    }

    //--------------------------------------------------------------------------
    // determine if A and/or B are value-agnostic
    //--------------------------------------------------------------------------

    // These 1st, 2nd, and pair operators are all handled by the flip, so if
    // flipxy is still true, all of these booleans will be false.
    bool op_is_first  = (mult_opcode == GB_FIRST_binop_code ) ;
    bool op_is_second = (mult_opcode == GB_SECOND_binop_code) ;
    bool op_is_pair   = (mult_opcode == GB_PAIR_binop_code) ;
    bool op_is_positional = GB_IS_BUILTIN_BINOP_CODE_POSITIONAL (mult_opcode) ;
    if (op_is_second || op_is_pair || op_is_positional || C_iso)
    { 
        // x is not used
        xcode = 0 ;
    }
    if (op_is_first  || op_is_pair || op_is_positional || C_iso)
    { 
        // y is not used
        ycode = 0  ;
    }
    bool A_is_pattern = (xcode == 0) ;
    bool B_is_pattern = (ycode == 0) ;

    //--------------------------------------------------------------------------
    // enumify the multiplier
    //--------------------------------------------------------------------------

    int mult_code = (mult_opcode - GB_USER_binop_code) & 0x3F ;

    //--------------------------------------------------------------------------
    // enumify the monoid
    //--------------------------------------------------------------------------

    ASSERT (add_opcode >= GB_USER_binop_code) ;
    ASSERT (add_opcode <= GB_BXNOR_binop_code) ;
    int add_code = (add_opcode - GB_USER_binop_code) & 0xF ;

    //--------------------------------------------------------------------------
    // enumify the types
    //--------------------------------------------------------------------------

    int acode = A_is_pattern ? 0 : atype->code ;   // 0 to 14
    int bcode = B_is_pattern ? 0 : btype->code ;   // 0 to 14
    int ccode = C_iso ? 0 : ctype->code ;          // 0 to 14

    int A_iso_code = (A_is_pattern || A->iso) ? 1 : 0 ;
    int B_iso_code = (B_is_pattern || B->iso) ? 1 : 0 ;
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
    int A_sparsity = GB_sparsity (A) ;
    int B_sparsity = GB_sparsity (B) ;

    int csparsity, msparsity, asparsity, bsparsity ;
    GB_enumify_sparsity (&csparsity, C_sparsity) ;
    GB_enumify_sparsity (&msparsity, M_sparsity) ;
    GB_enumify_sparsity (&asparsity, A_sparsity) ;
    GB_enumify_sparsity (&bsparsity, B_sparsity) ;

    //--------------------------------------------------------------------------
    // construct the semiring method_code
    //--------------------------------------------------------------------------

    // total method_code bits: 50 (13 hex digits): 14 bits to spare.

    (*method_code) =
                                               // range        bits
                // monoid (4 bits, 1 hex digit)
                GB_LSHIFT (add_code   , 48) |  // 0 to 13      4

                // C in, A, B iso properties, flipxy (1 hex digit)
                GB_LSHIFT (C_in_iso_cd, 47) |  // 0 or 1       1
                GB_LSHIFT (A_iso_code , 46) |  // 0 or 1       1
                GB_LSHIFT (B_iso_code , 45) |  // 0 or 1       1
                GB_LSHIFT (flipxy     , 44) |  // 0 to 1       1

                // multiplier, z = f(x,y) or f(y,x) (5 hex digits)
                // 2 bits unused here (42 and 43)
                GB_LSHIFT (mult_code  , 36) |  // 0 to 52      6
                GB_LSHIFT (zcode      , 32) |  // 0 to 14      4
                GB_LSHIFT (xcode      , 28) |  // 0 to 14      4
                GB_LSHIFT (ycode      , 24) |  // 0 to 14      4

                // mask (1 hex digit)
                GB_LSHIFT (mask_ecode , 20) |  // 0 to 13      4

                // types of C, A, and B (3 hex digits)
                GB_LSHIFT (ccode      , 16) |  // 0 to 14      4
                GB_LSHIFT (acode      , 12) |  // 0 to 14      4
                GB_LSHIFT (bcode      ,  8) |  // 0 to 14      4

                // sparsity structures of C, M, A, and B (2 hex digits)
                GB_LSHIFT (csparsity  ,  6) |  // 0 to 3       2
                GB_LSHIFT (msparsity  ,  4) |  // 0 to 3       2
                GB_LSHIFT (asparsity  ,  2) |  // 0 to 3       2
                GB_LSHIFT (bsparsity  ,  0) ;  // 0 to 3       2

}

