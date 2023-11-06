//------------------------------------------------------------------------------
// GxB_BinaryOp_new: create a new user-defined binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Create a new a binary operator: z = f (x,y).  The binary function signature
// must be void f (void *z, const void *x, const void *y), and then it must
// recast its input and output arguments internally as needed.

// If the function pointer is NULL, the function is compiled with the JIT.

#include "GB.h"
#include "GB_binop.h"
#include "GB_stringify.h"

GrB_Info GxB_BinaryOp_new
(
    GrB_BinaryOp *op_handle,        // handle for the new binary operator
    GxB_binary_function function,   // pointer to the binary function
    GrB_Type ztype,                 // type of output z
    GrB_Type xtype,                 // type of input x
    GrB_Type ytype,                 // type of input y
    const char *binop_name,         // name of the user function
    const char *binop_defn          // definition of the user function
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_BinaryOp_new (op, function, ztype, xtype, ytype"
        ", name, defn)") ;
    GB_RETURN_IF_NULL (op_handle) ;
    (*op_handle) = NULL ;
    GB_RETURN_IF_NULL_OR_FAULTY (ztype) ;
    GB_RETURN_IF_NULL_OR_FAULTY (xtype) ;
    GB_RETURN_IF_NULL_OR_FAULTY (ytype) ;

    //--------------------------------------------------------------------------
    // allocate the binary op
    //--------------------------------------------------------------------------

    size_t header_size ;
    GrB_BinaryOp op = GB_MALLOC (1, struct GB_BinaryOp_opaque, &header_size) ;
    if (op == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }
    op->header_size = header_size ;

    //--------------------------------------------------------------------------
    // create the binary op
    //--------------------------------------------------------------------------

    GrB_Info info = GB_binop_new (op, function, ztype, xtype, ytype,
        binop_name, binop_defn, GB_USER_binop_code) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        GB_FREE (&op, header_size) ;
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // create the function pointer, if NULL
    //--------------------------------------------------------------------------

    if (function == NULL)
    { 
        GB_BURBLE_START ("GxB_BinaryOp_new") ;
        void *user_function ;
        info = GB_user_op_jit (&user_function, (GB_Operator) op) ;
        if (info != GrB_SUCCESS)
        { 
            // unable to construct the function pointer
            GB_Op_free ((GB_Operator *) &op) ;
            return (GrB_NULL_POINTER) ;
        }
        op->binop_function = (GxB_binary_function) user_function ;
        GB_BURBLE_END ;
    }

    (*op_handle) = op ;
    return (GrB_SUCCESS) ;
}

