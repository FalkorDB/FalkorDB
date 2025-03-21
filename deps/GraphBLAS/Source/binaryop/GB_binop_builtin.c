//------------------------------------------------------------------------------
// GB_binop_builtin:  determine if a binary operator is built-in
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Determine if the binary operator is built-in and matches the types of its
// inputs.  Returns the opcodes and type codes of the op.  If this method
// returns true, the operator is a candidate for a switch factory.

#include "GB.h"
#include "binaryop/GB_binop.h"

bool GB_binop_builtin               // true if binary operator is builtin
(
    // inputs:
    const GrB_Type A_type,
    const bool A_is_pattern,        // true if only the pattern of A is used
    const GrB_Type B_type,
    const bool B_is_pattern,        // true if only the pattern of B is used
    const GrB_BinaryOp op,          // binary operator; may be NULL
    const bool flipxy,              // true if z=op(y,x), flipping x and y
    // outputs:
    GB_Opcode *opcode,              // opcode for the binary operator
    GB_Type_code *xcode,            // type code for x input
    GB_Type_code *ycode,            // type code for y input
    GB_Type_code *zcode             // type code for z output
)
{

    //--------------------------------------------------------------------------
    // check if the operator is builtin, with no typecasting
    //--------------------------------------------------------------------------

    ASSERT (op != NULL) ;
    (*opcode) = op->opcode ;
    (*xcode) = op->xtype->code ;
    (*ycode) = op->ytype->code ;
    (*zcode) = op->ztype->code ;

    if (flipxy)
    { 
        // For a semiring, GB_AxB_meta has already handled flipxy for built-in
        // semirings and operators that can be flipped.  If flipxy is still
        // true, the binary operator is not part of a built-in semiring.
        return (false) ;
    }

    ASSERT (GB_IS_BINARYOP_CODE (*opcode) ||
            GB_IS_INDEXBINARYOP_CODE (*opcode)) ;
    if (*opcode == GB_USER_binop_code || *opcode == GB_USER_idxbinop_code)
    { 
        // the binary operator is user-defined
        return (false) ;
    }

    bool op_is_builtin_positional =
        GB_IS_BUILTIN_BINOP_CODE_POSITIONAL (*opcode) ;

    // check if A matches the input to the operator
    if (!A_is_pattern && !op_is_builtin_positional)
    {
        if ((A_type != op->xtype) || (A_type->code >= GB_UDT_code))
        { 
            // A is a user-defined type, or its type does not match the input
            // to the operator
            return (false) ;
        }
    }

    // check if B matches the input to the operator
    if (!B_is_pattern && !op_is_builtin_positional)
    {
        if ((B_type != op->ytype) || (B_type->code >= GB_UDT_code))
        { 
            // B is a user-defined type, or its type does not match the input
            // to the operator
            return (false) ;
        }
    }

    //--------------------------------------------------------------------------
    // rename redundant boolean operators
    //--------------------------------------------------------------------------

    ASSERT ((*xcode) < GB_UDT_code) ;
    ASSERT ((*ycode) < GB_UDT_code) ;
    ASSERT ((*zcode) < GB_UDT_code) ;

    if ((*xcode) == GB_BOOL_code)
    { 
        // z = op(x,y) where both x and y are boolean.
        (*opcode) = GB_boolean_rename (*opcode) ;
    }

    return (true) ;
}

