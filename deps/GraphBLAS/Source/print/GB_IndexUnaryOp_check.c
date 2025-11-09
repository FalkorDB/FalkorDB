//------------------------------------------------------------------------------
// GB_IndexUnaryOp_check: check and print a index_unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "get_set/GB_get_set.h"

GrB_Info GB_IndexUnaryOp_check  // check a GraphBLAS index_unary operator
(
    const GrB_IndexUnaryOp op,  // GraphBLAS operator to print and check
    const char *name,       // name of the operator
    int pr,                 // print level
    FILE *f                 // file for output
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GBPR0 ("\n    GraphBLAS IndexUnaryOp: %s ", ((name != NULL) ? name : "")) ;

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
    if (!GB_IS_INDEXUNARYOP_CODE (opcode))
    { 
        GBPR0 ("    IndexUnaryOp has an invalid opcode\n") ;
        return (GrB_INVALID_OBJECT) ;
    }
    if (opcode == GB_USER_idxunop_code)
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
    GBPR0 ("z=%s(x,i,j,y)\n", op_name) ;

    if (op->idxunop_function == NULL)
    { 
        GBPR0 ("    IndexUnaryOp has a NULL function pointer\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    if (opcode == GB_USER_idxunop_code && name_len != actual_len)
    { 
        GBPR0 ("    IndexUnaryOp has an invalid name_len\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    // name given by GrB_set, or 'GrB_*' name for built-in operators
    const char *given_name = GB_op_name_get ((GB_Operator) op) ;
    if (given_name != NULL)
    { 
        GBPR0 ("    IndexUnaryOp given name: [%s]\n", given_name) ;
    }

    GrB_Info info ;

    info = GB_Type_check (op->ztype, "ztype", pr, f) ;
    if (info != GrB_SUCCESS)
    { 
        GBPR0 ("    IndexUnaryOp has an invalid ztype\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    if (op->xtype != NULL)
    {
        info = GB_Type_check (op->xtype, "xtype", pr, f) ;
        if (info != GrB_SUCCESS)
        { 
            GBPR0 ("    IndexUnaryOp has an invalid xtype\n") ;
            return (GrB_INVALID_OBJECT) ;
        }
    }

    info = GB_Type_check (op->ytype, "ytype", pr, f) ;
    if (info != GrB_SUCCESS)
    { 
        GBPR0 ("    IndexUnaryOp has an invalid ytype\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    if (op->defn != NULL)
    { 
        GBPR0 ("%s\n", op->defn) ;
    }

    return (GrB_SUCCESS) ;
}

