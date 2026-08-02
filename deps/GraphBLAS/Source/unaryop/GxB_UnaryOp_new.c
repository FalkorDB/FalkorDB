//------------------------------------------------------------------------------
// GxB_UnaryOp_new: create a new named unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// a unary operator: z = f (x).  The unary function signature must be
// void f (void *z, const void *x), and then it must recast its input and
// output arguments internally as needed.

// The op is allocated in header arena determined by the current Context.

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
    int header_arena = GB_Context_header_arena ( ) ;
    return (GxB_UnaryOp_new_arena (op_handle, function, ztype, xtype,
        unop_name, unop_defn, header_arena)) ;
}

