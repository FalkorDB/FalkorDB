//------------------------------------------------------------------------------
// GB_BinaryOp_compatible: check binary operator for type compatibility
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// check type compatibilty for C = op (A,B).  With typecasting: A is cast to
// op->xtype, B is cast to op->ytype, the operator is computed, and then the
// result of op->ztype is cast to C->type.

#include "GB.h"

GrB_Info GB_BinaryOp_compatible     // check for domain mismatch
(
    const GrB_BinaryOp op,          // binary operator to check
    const GrB_Type ctype,           // C must be compatible with op->ztype
    const GrB_Type atype,           // A must be compatible with op->xtype
    const GrB_Type btype,           // B must be compatible with op->ytype
    const GB_Type_code bcode,       // B may not have a type, just a code
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // ctype and btype may be NULL, but atype is never NULL
    ASSERT (op != NULL) ;
    ASSERT (atype != NULL) ;
    ASSERT (bcode <= GB_UDT_code) ;
    GB_Opcode opcode = op->opcode ;
    bool op_is_pair_or_positional = (opcode == GB_PAIR_binop_code) 
        || GB_OPCODE_IS_POSITIONAL (opcode) ;

    //--------------------------------------------------------------------------
    // first input A is cast into the type of op->xtype
    //--------------------------------------------------------------------------

    if (opcode == GB_SECOND_binop_code || op_is_pair_or_positional)
    { 
        // first input is unused, so A is always compatible
        ;
    }
    else if (!GB_Type_compatible (atype, op->xtype))
    { 
        GB_ERROR (GrB_DOMAIN_MISMATCH,
            "Incompatible type for z=%s(x,y):\n"
            "first input of type [%s]\n"
            "cannot be typecast to x input of type [%s]",
            op->name, atype->name, op->xtype->name) ;
    }

    //--------------------------------------------------------------------------
    // second input B is cast into the type of op->ytype
    //--------------------------------------------------------------------------

    if (opcode == GB_FIRST_binop_code || op_is_pair_or_positional)
    { 
        // second input is unused, so B is always compatible
        ;
    }
    else if (btype != NULL)
    {
        // B has a type
        if (!GB_Type_compatible (btype, op->ytype))
        { 
            GB_ERROR (GrB_DOMAIN_MISMATCH,
                "Incompatible type for z=%s(x,y):\n"
                "second input of type [%s]\n"
                "cannot be typecast to y input of type [%s]",
                op->name, btype->name, op->ytype->name) ;
        }
    }
    else
    {
        // B has a just a type code, not a type
        if (!GB_code_compatible (bcode, op->ytype->code))
        { 
            GB_ERROR (GrB_DOMAIN_MISMATCH,
                "Incompatible type for z=%s(x,y):\n"
                "second input of type [%s]\n"
                "cannot be typecast to y input of type [%s]",
                op->name, GB_code_string (bcode), op->ytype->name) ;
        }
    }

    //--------------------------------------------------------------------------
    // result of binary operator of op->ztype is cast to C
    //--------------------------------------------------------------------------

    if (!GB_Type_compatible (ctype, op->ztype))
    { 
        GB_ERROR (GrB_DOMAIN_MISMATCH,
            "Incompatible type for z=%s(x,y):\n"
            "operator output z of type [%s]\n"
            "cannot be typecast to result of type [%s]",
            op->name, op->ztype->name, ctype->name) ;
    }

    return (GrB_SUCCESS) ;
}

