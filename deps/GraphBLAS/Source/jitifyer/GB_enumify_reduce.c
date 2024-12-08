//------------------------------------------------------------------------------
// GB_enumify_reduce: enumerate a GrB_reduce problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

// Returns true if all types and operators are built-in.

void GB_enumify_reduce      // enumerate a GrB_reduce problem
(
    // output:
    uint64_t *method_code,  // unique encoding of the entire problem
    // input:
    GrB_Monoid monoid,      // the monoid to enumify
    GrB_Matrix A            // input matrix to monoid
)
{ 

    //--------------------------------------------------------------------------
    // get the monoid and type of A
    //--------------------------------------------------------------------------

    GrB_BinaryOp reduceop = monoid->op ;
    GrB_Type atype = A->type ;
    GrB_Type ztype = reduceop->ztype ;
    GB_Opcode reduce_opcode  = reduceop->opcode ;
    // these must always be true for any monoid:
    ASSERT (reduceop->xtype == reduceop->ztype) ;
    ASSERT (reduceop->ytype == reduceop->ztype) ;

    //--------------------------------------------------------------------------
    // rename redundant boolean operators
    //--------------------------------------------------------------------------

    // consider z = op(x,y) where both x and y are boolean:
    // DIV becomes FIRST
    // RDIV becomes SECOND
    // MIN and TIMES become LAND
    // MAX and PLUS become LOR
    // NE, ISNE, RMINUS, and MINUS become LXOR
    // ISEQ becomes EQ
    // ISGT becomes GT
    // ISLT becomes LT
    // ISGE becomes GE
    // ISLE becomes LE

    GB_Type_code zcode = ztype->code ;
    if (zcode == GB_BOOL_code)
    { 
        // rename the monoid
        reduce_opcode = GB_boolean_rename (reduce_opcode) ;
    }

    //--------------------------------------------------------------------------
    // enumify the monoid
    //--------------------------------------------------------------------------

    ASSERT (reduce_opcode >= GB_USER_binop_code) ;
    ASSERT (reduce_opcode <= GB_BXNOR_binop_code) ;
    int red_code = (reduce_opcode - GB_USER_binop_code) & 0xF ;

    const char *a = NULL, *cuda_type = NULL ;
    bool user_monoid_atomically = false ;
    bool has_cheeseburger = GB_enumify_cuda_atomic (&a,
        &user_monoid_atomically, &cuda_type,
        monoid, reduce_opcode, ztype->size, zcode) ;
    int cheese = (has_cheeseburger) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // enumify the type and sparsity structure of A
    //--------------------------------------------------------------------------

    int acode = atype->code ;   // 0 to 14
    int asparsity ;
    GB_enumify_sparsity (&asparsity, GB_sparsity (A)) ;
    int azombies = (A->nzombies > 0) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // construct the reduction method_code
    //--------------------------------------------------------------------------

    // total method_code bits: 17 (5 hex digits)

    (*method_code) = 
                                               // range        bits
                // monoid: 5 bits (2 hex digits)
                GB_LSHIFT (cheese     , 16) |  // 0 to 1       1
                GB_LSHIFT (red_code   , 12) |  // 0 to 13      4

                // type of the monoid: 1 hex digit
                GB_LSHIFT (zcode      ,  8) |  // 0 to 14      4

                // type of A: 1 hex digit
                GB_LSHIFT (acode      ,  4) |  // 0 to 14      4

                // sparsity structure and zombies: 1 hex digit
                // unused bit            3                     1
                // zombies
                GB_LSHIFT (azombies   ,  2) |  // 0 to 1       1
                // sparsity structure of A
                GB_LSHIFT (asparsity  ,  0) ;  // 0 to 3       2

}

