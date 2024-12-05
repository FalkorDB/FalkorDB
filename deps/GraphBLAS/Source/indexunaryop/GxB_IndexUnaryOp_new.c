//------------------------------------------------------------------------------
// GxB_IndexUnaryOp_new: create a new user-defined index_unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Create a new a index_unary operator: z = f (x,i,j,thunk).  The
// index_unary function signature must be:

// void f (void *z, const void *x, uint64_t i, uint64_t j, const void *thunk)

// and then it must recast its inputs (x and thunk) and output (z) arguments
// internally as needed.  When used with a GrB_Vector, j is zero.

// If the function pointer is NULL, the function is compiled with the JIT.

#include "GB.h"
#include "jitifyer/GB_stringify.h"

GrB_Info GxB_IndexUnaryOp_new   // create a named user-created IndexUnaryOp
(
    GrB_IndexUnaryOp *op_handle,    // handle for the new IndexUnary operator
    GxB_index_unary_function function,    // pointer to index_unary function
    GrB_Type ztype,                 // type of output z
    GrB_Type xtype,                 // type of input x (the A(i,j) entry)
    GrB_Type ytype,                 // type of input y (the scalar)
    const char *idxop_name,         // name of the user function
    const char *idxop_defn          // definition of the user function
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_IndexUnaryOp_new (op, function, ztype, xtype, ytype"
        ", name, defn)") ;
    GB_RETURN_IF_NULL (op_handle) ;
    (*op_handle) = NULL ;
    GB_RETURN_IF_NULL_OR_FAULTY (ztype) ;
    GB_RETURN_IF_NULL_OR_FAULTY (xtype) ;
    GB_RETURN_IF_NULL_OR_FAULTY (ytype) ;

    //--------------------------------------------------------------------------
    // allocate the index_unary op
    //--------------------------------------------------------------------------

    size_t header_size ;
    GrB_IndexUnaryOp op = GB_CALLOC (1, struct GB_IndexUnaryOp_opaque,
        &header_size) ;
    if (op == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }
    op->header_size = header_size ;

    //--------------------------------------------------------------------------
    // initialize the index_unary operator
    //--------------------------------------------------------------------------

    op->magic = GB_MAGIC ;
    op->user_name = NULL ;
    op->user_name_size = 0 ;
    op->ztype = ztype ;
    op->xtype = xtype ;
    op->ytype = ytype ;      // thunk type

    op->unop_function = NULL ;
    op->idxunop_function = function ;
    op->binop_function = NULL ;

    op->opcode = GB_USER_idxunop_code ;

    //--------------------------------------------------------------------------
    // get the index_unary op name and defn
    //--------------------------------------------------------------------------

    // the index_unary op is JIT'able only if all its types are jitable
    bool jitable =
        (ztype->hash != UINT64_MAX) &&
        (xtype->hash != UINT64_MAX) &&
        (ytype->hash != UINT64_MAX) ;

    GrB_Info info = GB_op_name_and_defn (
        // output:
        op->name, &(op->name_len), &(op->hash), &(op->defn), &(op->defn_size),
        // input:
        idxop_name, idxop_defn, true, jitable) ;
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
        GB_BURBLE_START ("GxB_IndexUnaryOp_new") ;
        void *user_function ;
        info = GB_user_op_jit (&user_function, (GB_Operator) op) ;
        if (info != GrB_SUCCESS)
        {
            // unable to construct the function pointer
            GB_Op_free ((GB_Operator *) &op) ;
            // If the JIT fails, it returns GrB_NO_VALUE or GxB_JIT_ERROR.
            // Convert GrB_NO_VALUE to GrB_NULL_POINTER (the function is NULL
            // and cannot be compiled by the JIT).
            return (info == GrB_NO_VALUE ? GrB_NULL_POINTER : info) ;
        }
        op->idxunop_function = (GxB_index_unary_function) user_function ;
        GB_BURBLE_END ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_INDEXUNARYOP_OK (op, "new user-defined index_unary op", GB0) ;
    (*op_handle) = op ;
    return (GrB_SUCCESS) ;
}

