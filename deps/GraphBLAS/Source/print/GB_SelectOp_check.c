//------------------------------------------------------------------------------
// GB_SelectOp_check: check and print a select operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GB_SelectOp_check  // check a GraphBLAS select operator
(
    const GxB_SelectOp op,  // GraphBLAS operator to print and check
    const char *name,       // name of the operator
    int pr,                 // print level
    FILE *f                 // file for output
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GBPR0 ("\n    GraphBLAS SelectOp: %s: ", ((name != NULL) ? name : "")) ;

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
    if (!GB_IS_SELECTOP_CODE (opcode))
    {
        GBPR0 ("    SelectOp has an invalid opcode\n") ;
        return (GrB_INVALID_OBJECT) ;
    }
    GBPR0 ("C=%s(A,k)\n", op->name) ;

    if (op->xtype != NULL)
    {
        GrB_Info info = GB_Type_check (op->xtype, "xtype", pr, f) ;
        if (info != GrB_SUCCESS)
        {
            GBPR0 ("    SelectOp has an invalid xtype\n") ;
            return (GrB_INVALID_OBJECT) ;
        }
    }

    if (op->ytype != NULL)
    {
        GrB_Info info = GB_Type_check (op->ytype, "thunk type", pr, f) ;
        if (info != GrB_SUCCESS)
        {
            GBPR0 ("    SelectOp has an invalid thunk type\n") ;
            return (GrB_INVALID_OBJECT) ;
        }
    }

    return (GrB_SUCCESS) ;
}

