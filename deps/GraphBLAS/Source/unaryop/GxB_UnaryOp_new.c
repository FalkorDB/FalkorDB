//------------------------------------------------------------------------------
// GxB_UnaryOp_new: create a new named unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// a unary operator: z = f (x).  The unary function signature must be
// void f (void *z, const void *x), and then it must recast its input and
// output arguments internally as needed.

#include "GB.h"
#include "unaryop/GB_unop.h"
#include "jitifyer/GB_stringify.h"

GrB_Info GxB_UnaryOp_new            // create a new user-defined unary operator
(
    GrB_UnaryOp *op_handle,         // handle for the new unary operator
    GxB_unary_function function,    // pointer to the unary function
    GrB_Type ztype,                 // type of output z
    GrB_Type xtype,                 // type of input x
    const char *unop_name,          // name of the user function
    const char *unop_defn           // definition of the user function
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL (op_handle) ;
    (*op_handle) = NULL ;
    GB_RETURN_IF_NULL_OR_FAULTY (ztype) ;
    GB_RETURN_IF_NULL_OR_FAULTY (xtype) ;

    //--------------------------------------------------------------------------
    // create the unary op
    //--------------------------------------------------------------------------

    // allocate the unary operator
    size_t header_size ;
    GrB_UnaryOp op = GB_CALLOC_MEMORY (1, sizeof (struct GB_UnaryOp_opaque),
        &header_size) ;
    if (op == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    op->header_size = header_size ;

    GrB_Info info = GB_unop_new (op, function, ztype, xtype, unop_name,
        unop_defn, GB_USER_unop_code) ;

    if (info != GrB_SUCCESS)
    { 
        // out of memory
        GB_FREE_MEMORY (&op, header_size) ;
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // create the function pointer, if NULL
    //--------------------------------------------------------------------------

    if (function == NULL)
    { 
        GB_BURBLE_START ("GxB_UnaryOp_new") ;
        void *user_function ;
        info = GB_user_op_jit (&user_function, (GB_Operator) op) ;
        if (info != GrB_SUCCESS)
        { 
            // unable to construct the function pointer
            GB_Op_free ((GB_Operator *) &op) ;
            // If the JIT fails, it returns GrB_NO_VALUE or GxB_JIT_ERROR,
            // Convert GrB_NO_VALUE to GrB_NULL_POINTER (the function is NULL
            // and cannot be compiled by the JIT).
            return (info == GrB_NO_VALUE ? GrB_NULL_POINTER : info) ;
        }
        #include "include/GB_pedantic_disable.h"
        op->unop_function = (GxB_unary_function) user_function ;
        GB_BURBLE_END ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_UNARYOP_OK (op, "new user-defined unary op", GB0) ;
    (*op_handle) = op ;
    return (GrB_SUCCESS) ;
}

