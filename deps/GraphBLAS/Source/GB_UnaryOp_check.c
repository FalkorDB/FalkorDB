//------------------------------------------------------------------------------
// GB_UnaryOp_check: check and print a unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GB_UnaryOp_check   // check a GraphBLAS unary operator
(
    const GrB_UnaryOp op,   // GraphBLAS operator to print and check
    const char *name,       // name of the operator
    int pr,                 // print level
    FILE *f                 // file for output
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GBPR0 ("\n    GraphBLAS UnaryOp: %s ", ((name != NULL) ? name : "")) ;

    if (op == NULL)
    { 
        GBPR0 ("NULL\n") ;
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // check object
    //--------------------------------------------------------------------------

    GB_CHECK_MAGIC (op) ;
    GB_Opcode opcode = op->opcode ;
    if (!GB_IS_UNARYOP_CODE (opcode))
    { 
        GBPR0 ("    UnaryOp has an invalid opcode\n") ;
        return (GrB_INVALID_OBJECT) ;
    }
    if (opcode == GB_USER_unop_code)
    { 
        GBPR0 ("(user-defined): ") ;
    }
    else
    { 
        GBPR0 ("(built-in): ") ;
    }
    int32_t name_len = op->name_len ;
    int32_t actual_len = (int32_t) strlen (op->name) ;
    char *op_name = (actual_len > 0) ? op->name : "f" ;
    GBPR0 ("z=%s(x)\n", op_name) ;

    bool op_is_positional = GB_OPCODE_IS_POSITIONAL (opcode) ;
    bool op_is_one = (opcode == GB_ONE_unop_code) ;
    bool op_is_identity = (opcode == GB_IDENTITY_unop_code) ;

    if (!op_is_positional && op->unop_function == NULL && !(op_is_identity))
    { 
        GBPR0 ("    UnaryOp has a NULL function pointer\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    if (opcode == GB_USER_unop_code && name_len != actual_len)
    { 
        GBPR0 ("    UnaryOp has an invalid name_len\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    GrB_Info info ;

    info = GB_Type_check (op->ztype, "ztype", pr, f) ;
    if (info != GrB_SUCCESS)
    { 
        GBPR0 ("    UnaryOp has an invalid ztype\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    if (!op_is_positional && !op_is_one)
    {
        info = GB_Type_check (op->xtype, "xtype", pr, f) ;
        if (info != GrB_SUCCESS)
        { 
            GBPR0 ("    UnaryOp has an invalid xtype\n") ;
            return (GrB_INVALID_OBJECT) ;
        }
    }

    if (op->defn != NULL)
    { 
        GBPR0 ("%s\n", op->defn) ;
    }

    return (GrB_SUCCESS) ;
}

