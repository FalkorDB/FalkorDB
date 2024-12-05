//------------------------------------------------------------------------------
// GB_enumify_sort: enumerate a GxB_sort problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is sparse or hypersparse, but the algorithm doesn't access C->h, and works
// identically for both cases.  So the JIT kernel can treat C as if sparse.

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_enumify_sort        // enumerate a GxB_sort problem
(
    // output:
    uint64_t *method_code,  // unique encoding of the entire operation
    // input:
    GrB_Matrix C,           // matrix to sort
    // comparator op:
    GrB_BinaryOp binaryop   // the binary operator for the comparator
)
{

    //--------------------------------------------------------------------------
    // get the type of C
    //--------------------------------------------------------------------------

    GrB_Type ctype = C->type ;
    int ccode = ctype->code ;           // 1 to 14

    //--------------------------------------------------------------------------
    // get the type of X and the opcode
    //--------------------------------------------------------------------------

    ASSERT (binaryop != NULL) ;

    GB_Opcode opcode = binaryop->opcode ;
    GB_Type_code xcode = binaryop->xtype->code ;

    // the comparator op, z=f(x,y) must have a ztype of boolean,
    // and the x and y types must match
    ASSERT (binaryop->xtype == binaryop->ytype) ;
    ASSERT (binaryop->ztype == GrB_BOOL) ;

    if (xcode == GB_BOOL_code)  // && (ycode == GB_BOOL_code)
    { 
        // rename the operator
        opcode = GB_boolean_rename (opcode) ;
    }

    //--------------------------------------------------------------------------
    // enumify the binary operator
    //--------------------------------------------------------------------------

    int binop_code = (opcode - GB_USER_binop_code) & 0x3F ;

    //--------------------------------------------------------------------------
    // construct the sort method_code
    //--------------------------------------------------------------------------

    // total method_code bits: 14 (4 hex digits)

    (*method_code) =
                                               // range        bits
                // binaryop, z = f(x,y) (3 hex digits)
                GB_LSHIFT (binop_code , 12) |  // 0 to 52      6
                GB_LSHIFT (xcode      ,  8) |  // 1 to 14      4

                // type of C (1 hex digit)
                GB_LSHIFT (ccode      ,  0) ;  // 1 to 14      4
}

