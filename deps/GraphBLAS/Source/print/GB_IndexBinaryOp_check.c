//------------------------------------------------------------------------------
// GB_IndexBinaryOp_check: check and print a index_binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "get_set/GB_get_set.h"

GrB_Info GB_IndexBinaryOp_check  // check a GraphBLAS index_binary operator
(
    const GxB_IndexBinaryOp op,  // GraphBLAS operator to print and check
    const char *name,       // name of the operator
    int pr,                 // print level
    FILE *f                 // file for output
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GBPR0 ("\n    GraphBLAS IndexBinaryOp: %s ", ((name != NULL) ? name : "")) ;

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
    if (!GB_IS_INDEXBINARYOP_CODE (opcode))
    { 
        GBPR0 ("    IndexBinaryOp has an invalid opcode\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    // only user-defined IndexBinaryOps exist
    ASSERT (opcode == GB_USER_idxbinop_code) ;
    GBPR0 ("(user-defined):\n    ") ;

    int32_t name_len = op->name_len ;
    int32_t actual_len = (int32_t) strlen (op->name) ;
    char *op_name = (actual_len > 0) ? op->name : "f" ;
    GBPR0 ("z=%s(x,ix,jx,y,iy,jy,theta)\n", op_name) ;

    // name given by GrB_set, or 'GrB_*' name for built-in operators
    const char *given_name = GB_op_name_get ((GB_Operator) op) ;
    if (given_name != NULL)
    { 
        GBPR0 ("    IndexBinaryOp given name: [%s]\n", given_name) ;
    }

    if (op->idxbinop_function == NULL)
    { 
        GBPR0 ("    IndexBinaryOp has a NULL function pointer\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    if (opcode == GB_USER_idxbinop_code && name_len != actual_len)
    { 
        GBPR0 ("    IndexBinaryOp has an invalid name_len\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    GrB_Info info ;

    info = GB_Type_check (op->ztype, "ztype", pr, f) ;
    if (info != GrB_SUCCESS)
    { 
        GBPR0 ("    IndexBinaryOp has an invalid ztype\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    info = GB_Type_check (op->xtype, "xtype", pr, f) ;
    if (info != GrB_SUCCESS)
    { 
        GBPR0 ("    IndexBinaryOp has an invalid xtype\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    info = GB_Type_check (op->ytype, "ytype", pr, f) ;
    if (info != GrB_SUCCESS)
    { 
        GBPR0 ("    IndexBinaryOp has an invalid ytype\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    info = GB_Type_check (op->theta_type, "theta_type", pr, f) ;
    if (info != GrB_SUCCESS)
    { 
        GBPR0 ("    IndexBinaryOp has an invalid theta_type\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    if (op->defn != NULL)
    { 
        GBPR0 ("%s\n", op->defn) ;
    }

    return (GrB_SUCCESS) ;
}

