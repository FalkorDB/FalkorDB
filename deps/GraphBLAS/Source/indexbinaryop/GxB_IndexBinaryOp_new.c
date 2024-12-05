//------------------------------------------------------------------------------
// GxB_IndexBinaryOp_new: create a new user-defined index_binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Create a new a index_binary operator: z = f (x,ix,jx, y,iy,jy, theta).  The
// index_binary function signature must be:

// void f (void *z,
//      const void *x, uint64_t ix, uint64_t jx,
//      const void *y, uint64_t iy, uint64_t jy,
//      const void *theta)

// and then it must recast its inputs (x and theta) and output (z) arguments
// internally as needed.  When used with GrB_Vectors, jx and jy are zero.

// If the function pointer is NULL, the function is compiled with the JIT.

#include "GB.h"
#include "jitifyer/GB_stringify.h"

GrB_Info GxB_IndexBinaryOp_new
(
    GxB_IndexBinaryOp *op_handle,   // handle for the new index binary operator
    GxB_index_binary_function function, // pointer to the index binary function
    GrB_Type ztype,                 // type of output z
    GrB_Type xtype,                 // type of input x
    GrB_Type ytype,                 // type of input y
    GrB_Type theta_type,            // type of input theta
    const char *idxop_name,         // name of the user function
    const char *idxop_defn          // definition of the user function
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_IndexBinaryOp_new (op, function, ztype, xtype, ytype"
        ", theta_type, name, defn)") ;
    GB_RETURN_IF_NULL (op_handle) ;
    (*op_handle) = NULL ;
    GB_RETURN_IF_NULL_OR_FAULTY (ztype) ;
    GB_RETURN_IF_NULL_OR_FAULTY (xtype) ;
    GB_RETURN_IF_NULL_OR_FAULTY (ytype) ;
    GB_RETURN_IF_NULL_OR_FAULTY (theta_type) ;

    //--------------------------------------------------------------------------
    // allocate the index_binary op
    //--------------------------------------------------------------------------

    size_t header_size ;
    GxB_IndexBinaryOp op = GB_CALLOC (1, struct GB_IndexBinaryOp_opaque,
        &header_size) ;
    if (op == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }
    op->header_size = header_size ;

    //--------------------------------------------------------------------------
    // initialize the index_binary operator
    //--------------------------------------------------------------------------

    op->magic = GB_MAGIC ;
    op->user_name = NULL ;
    op->user_name_size = 0 ;
    op->ztype = ztype ;
    op->xtype = xtype ;
    op->ytype = ytype ;
    op->theta_type = theta_type ;

    op->unop_function = NULL ;
    op->idxunop_function = NULL ;
    op->binop_function = NULL ;
    op->idxbinop_function = function ;

    op->opcode = GB_USER_idxbinop_code ;

    //--------------------------------------------------------------------------
    // get the index_binary op name and defn
    //--------------------------------------------------------------------------

    // the index_binary op is JIT'able only if all its types are jitable
    bool jitable =
        (ztype->hash != UINT64_MAX) &&
        (xtype->hash != UINT64_MAX) &&
        (ytype->hash != UINT64_MAX) &&
        (theta_type->hash != UINT64_MAX) ;

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
        GB_BURBLE_START ("GxB_IndexBinaryOp_new") ;
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
        op->idxbinop_function = (GxB_index_binary_function) user_function ;
        GB_BURBLE_END ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_INDEXBINARYOP_OK (op, "new user-defined index_binary op", GB0) ;
    (*op_handle) = op ;
    return (GrB_SUCCESS) ;
}

