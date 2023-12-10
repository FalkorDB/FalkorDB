//------------------------------------------------------------------------------
// GB_unop_new: create a new named unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// a unary operator: z = f (x).  The unary function signature must be
// void f (void *z, const void *x), and then it must recast its input and
// output arguments internally as needed.

// The unary op header is allocated by the caller, and passed in uninitialized.

#include "GB.h"
#include "GB_unop.h"

GrB_Info GB_unop_new
(
    GrB_UnaryOp op,                 // new unary operator
    GxB_unary_function function,    // unary function (may be NULL)
    GrB_Type ztype,                 // type of output z
    GrB_Type xtype,                 // type of input x
    const char *unop_name,          // name of the user function
    const char *unop_defn,          // definition of the user function
    const GB_Opcode opcode          // opcode for the function
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (op != NULL) ;
    ASSERT (ztype != NULL) ;
    ASSERT (xtype != NULL) ;
    ASSERT (GB_IS_UNARYOP_CODE (opcode)) ;

    //--------------------------------------------------------------------------
    // initialize the unary operator
    //--------------------------------------------------------------------------

    op->magic = GB_MAGIC ;
    op->user_name = NULL ;
    op->user_name_size = 0 ;
    op->xtype = xtype ;
    op->ztype = ztype ;
    op->ytype = NULL ;

    op->unop_function = function ;      // NULL for IDENTITY_UDT operator
    op->idxunop_function = NULL ;
    op->binop_function = NULL ;

    op->opcode = opcode ;

    //--------------------------------------------------------------------------
    // get the unary op name and defn
    //--------------------------------------------------------------------------

    // the unary op is JIT'able only if all its types are jitable
    bool jitable =
        (ztype->hash != UINT64_MAX) &&
        (xtype->hash != UINT64_MAX) ;

    return (GB_op_name_and_defn (
        // output:
        op->name, &(op->name_len), &(op->hash), &(op->defn), &(op->defn_size),
        // input:
        unop_name, unop_defn, opcode == GB_USER_unop_code, jitable)) ;
}

