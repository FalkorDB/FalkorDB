//------------------------------------------------------------------------------
// GB_boolean_rename_op: rename a boolean operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// If the user requests the creation of a monoid based on a duplicate
// built-in binary operator, the unique boolean operator is used instead.
// See also GB_boolean_rename, which does this for opcodes, not operators.
// This is done before the operator is checked, so that any error messages
// reflect the renaming.

#include "GB.h"
#include "GB_binop.h"

GrB_BinaryOp GB_boolean_rename_op   // return renamed op
(
    const GrB_BinaryOp op           // op to rename
)
{

    if (op == GrB_DIV_BOOL)
    { 
        // FIRST and DIV are the same for boolean:
        return (GrB_FIRST_BOOL) ;
    }
    if (op == GxB_RDIV_BOOL)
    { 
        // SECOND and RDIV are the same for boolean:
        return (GrB_SECOND_BOOL) ;
    }
    if (op == GrB_MIN_BOOL || op == GrB_TIMES_BOOL)
    { 
        // MIN, TIMES, and LAND are the same for boolean:
        return (GrB_LAND) ;
    }
    if (op == GrB_MAX_BOOL || op == GrB_PLUS_BOOL)
    { 
        // MAX, PLUS, and OR are the same for boolean:
        return (GrB_LOR) ;
    }
    if (op == GxB_ISNE_BOOL || op == GrB_NE_BOOL || op == GrB_MINUS_BOOL
        || op == GxB_RMINUS_BOOL)
    { 
        // ISNE, NE, MINUS, RMINUS, and XOR are the same for boolean:
        return (GrB_LXOR) ;
    }
    if (op == GxB_ISEQ_BOOL || op == GrB_LXNOR)
    { 
        // LXNOR, ISEQ, EQ are the same for boolean:
        return (GrB_EQ_BOOL) ;
    }
    if (op == GxB_ISGT_BOOL)
    { 
        // ISGT, GT are the same for boolean:
        return (GrB_GT_BOOL) ;
    }
    if (op == GxB_ISLT_BOOL)
    { 
        // ISLT, LT are the same for boolean:
        return (GrB_LT_BOOL) ;
    }
    if (op == GxB_ISGE_BOOL || op == GxB_POW_BOOL)
    { 
        // POW, ISGE, GE are the same for boolean:
        return (GrB_GE_BOOL) ;
    }
    if (op == GxB_ISLE_BOOL)
    { 
        // ISLE, LE are the same for boolean:
        return (GrB_LE_BOOL) ;
    }

    // operator is not changed
    return (op) ;
}

