//------------------------------------------------------------------------------
// GxB_BinaryOp_new: create a new user-defined binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Create a new a binary operator: z = f (x,y).  The binary function signature
// must be void f (void *z, const void *x, const void *y), and then it must
// recast its input and output arguments internally as needed.

// If the function pointer is NULL, the function is compiled with the JIT.

// The op is allocated in the header arena determined by the current Context.

#include "GB.h"
#include "binaryop/GB_binop.h"
#include "jitifyer/GB_stringify.h"

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
    int header_arena = GB_Context_header_arena ( ) ;
    return (GxB_BinaryOp_new_arena (op_handle, function, ztype, xtype, ytype,
        binop_name, binop_defn, header_arena)) ;
}

