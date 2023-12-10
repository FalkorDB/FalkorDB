//------------------------------------------------------------------------------
// GB_enumify_reduce: enumerate a GrB_reduce problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

// Returns true if all types and operators are built-in.

void GB_enumify_reduce      // enumerate a GrB_reduce problem
(
    // output:
    uint64_t *rcode,        // unique encoding of the entire problem
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

    int red_ecode, id_ecode, term_ecode ;
    GB_enumify_monoid (&red_ecode, &id_ecode, &term_ecode, reduce_opcode,
        zcode) ;

    const char *a = NULL, *cuda_type = NULL ;
    bool user_monoid_atomically = false ;
    bool has_cheeseburger = GB_enumify_cuda_atomic (&a,
        &user_monoid_atomically, &cuda_type,
        monoid, red_ecode, ztype->size, zcode) ;
    int cheese = (has_cheeseburger) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // enumify the type and sparsity structure of A
    //--------------------------------------------------------------------------

    int acode = atype->code ;   // 0 to 14
    int asparsity ;
    GB_enumify_sparsity (&asparsity, GB_sparsity (A)) ;
    int azombies = (A->nzombies > 0) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // construct the reduction rcode
    //--------------------------------------------------------------------------

    // total rcode bits: 28 (7 hex digits)

    (*rcode) = 
                                               // range        bits
                // monoid: 16 bits (4 hex digits)
                GB_LSHIFT (cheese     , 27) |  // 0 to 1       1
                GB_LSHIFT (red_ecode  , 22) |  // 0 to 22      5
                GB_LSHIFT (id_ecode   , 17) |  // 0 to 31      5
                GB_LSHIFT (term_ecode , 12) |  // 0 to 31      5

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

