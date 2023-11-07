//------------------------------------------------------------------------------
// GB_Operator_check: check and print any operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GB_Operator_check  // check a GraphBLAS operator
(
    const GB_Operator op,   // GraphBLAS operator to print and check
    const char *name,       // name of the operator
    int pr,                 // print level
    FILE *f                 // file for output
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (op == NULL)
    { 
        GBPR0 ("\n    GraphBLAS Operator: %s: NULL\n",
            ((name != NULL) ? name : "")) ;
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // check the operator
    //--------------------------------------------------------------------------

    GB_Opcode opcode = op->opcode ;

    if (GB_IS_UNARYOP_CODE (opcode))
    { 
        return (GB_UnaryOp_check ((GrB_UnaryOp) op, name, pr, f)) ;
    }
    else if (GB_IS_BINARYOP_CODE (opcode))
    { 
        return (GB_BinaryOp_check ((GrB_BinaryOp) op, name, pr, f)) ;
    }
    else if (GB_IS_INDEXUNARYOP_CODE (opcode))
    { 
        return (GB_IndexUnaryOp_check ((GrB_IndexUnaryOp) op, name, pr, f)) ;
    }
    else if (GB_IS_SELECTOP_CODE (opcode))
    { 
        return (GB_SelectOp_check ((GxB_SelectOp) op, name, pr, f)) ;
    }
    else
    { 
        GBPR0 ("\n    GraphBLAS Operator: %s: invalid opcode\n",
            ((name != NULL) ? name : "")) ;
        return (GrB_INVALID_OBJECT) ;    // bad opcode
    }
}

