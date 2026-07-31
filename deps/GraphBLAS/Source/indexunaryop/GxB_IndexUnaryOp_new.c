//------------------------------------------------------------------------------
// GxB_IndexUnaryOp_new: create a new user-defined index_unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Create a new a index_unary operator: z = f (x,i,j,thunk).  The
// index_unary function signature must be:

// void f (void *z, const void *x, uint64_t i, uint64_t j, const void *thunk)

// and then it must recast its inputs (x and thunk) and output (z) arguments
// internally as needed.  When used with a GrB_Vector, j is zero.

// If the function pointer is NULL, the function is compiled with the JIT.

// The op is allocated in header arena determined by the current Context.

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
    int header_arena = GB_Context_header_arena ( ) ;
    return (GxB_IndexUnaryOp_new_arena (op_handle, function, ztype, xtype,
        ytype, idxop_name, idxop_defn, header_arena)) ;
}

