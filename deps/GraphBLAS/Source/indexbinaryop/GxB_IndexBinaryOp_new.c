//------------------------------------------------------------------------------
// GxB_IndexBinaryOp_new: create a new user-defined index_binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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

// The op is allocated in header arena determined by the current Context.

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
    int header_arena = GB_Context_header_arena ( ) ;
    return (GxB_IndexBinaryOp_new_arena (op_handle, function, ztype, xtype,
        ytype, theta_type, idxop_name, idxop_defn, header_arena)) ;
}

